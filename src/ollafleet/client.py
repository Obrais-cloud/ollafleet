"""Async client for querying multiple Ollama instances."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import anyio
import httpx


@dataclass
class ModelInfo:
    name: str
    size: int  # bytes
    parameter_size: str
    quantization: str
    modified_at: str
    digest: str

    @property
    def size_gb(self) -> float:
        return self.size / (1024**3)


@dataclass
class NodeStatus:
    name: str
    host: str
    tags: list[str]
    online: bool
    version: str | None = None
    models: list[ModelInfo] = field(default_factory=list)
    running: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class NodeConfig:
    name: str
    host: str
    tags: list[str] = field(default_factory=list)


class OllamaNode:
    """Client for a single Ollama instance."""

    def __init__(self, config: NodeConfig, timeout: float = 15.0):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.host,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    async def close(self):
        await self.client.aclose()

    async def ping(self) -> tuple[bool, float, str | None]:
        start = time.monotonic()
        try:
            resp = await self.client.get("/api/version")
            resp.raise_for_status()
            latency = (time.monotonic() - start) * 1000
            version = resp.json().get("version", "unknown")
            return True, latency, version
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return False, latency, str(e)

    async def list_models(self) -> list[ModelInfo]:
        resp = await self.client.get("/api/tags")
        resp.raise_for_status()
        models = []
        for m in resp.json().get("models", []):
            details = m.get("details", {})
            models.append(ModelInfo(
                name=m["name"],
                size=m.get("size", 0),
                parameter_size=details.get("parameter_size", "?"),
                quantization=details.get("quantization_level", "?"),
                modified_at=m.get("modified_at", ""),
                digest=m.get("digest", "")[:12],
            ))
        return models

    async def list_running(self) -> list[dict]:
        try:
            resp = await self.client.get("/api/ps")
            resp.raise_for_status()
            return resp.json().get("models", [])
        except Exception:
            return []

    async def pull_model(self, model: str) -> AsyncIterator[dict]:
        async with self.client.stream(
            "POST", "/api/pull", json={"name": model}, timeout=600.0
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    import json
                    yield json.loads(line)

    async def chat_stream(self, model: str, messages: list[dict]) -> AsyncIterator[str]:
        async with self.client.stream(
            "POST",
            "/api/chat",
            json={"model": model, "messages": messages, "stream": True},
            timeout=300.0,
        ) as resp:
            resp.raise_for_status()
            import json
            async for line in resp.aiter_lines():
                if line.strip():
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        return

    async def generate_once(self, model: str, prompt: str) -> tuple[str, float]:
        """Single non-streaming generate. Returns (response, tok/s)."""
        start = time.monotonic()
        resp = await self.client.post(
            "/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=300.0,
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.monotonic() - start
        response_text = data.get("response", "")
        eval_count = data.get("eval_count", 0)
        eval_duration = data.get("eval_duration", 0)
        tok_s = (eval_count / (eval_duration / 1e9)) if eval_duration > 0 else 0
        return response_text, tok_s

    async def status(self) -> NodeStatus:
        online, latency, version_or_err = await self.ping()
        node_status = NodeStatus(
            name=self.config.name,
            host=self.config.host,
            tags=self.config.tags,
            online=online,
            latency_ms=latency,
        )
        if not online:
            node_status.error = version_or_err
            return node_status

        node_status.version = version_or_err
        try:
            node_status.models = await self.list_models()
        except Exception:
            pass
        try:
            node_status.running = await self.list_running()
        except Exception:
            pass
        return node_status


class Fleet:
    """Manages multiple OllamaNode instances."""

    def __init__(self, nodes: list[NodeConfig], timeout: float = 15.0):
        self.nodes = [OllamaNode(cfg, timeout) for cfg in nodes]

    async def close(self):
        for node in self.nodes:
            await node.close()

    async def status_all(self) -> list[NodeStatus]:
        results: list[NodeStatus] = [None] * len(self.nodes)  # type: ignore

        async def _check(i: int, node: OllamaNode):
            results[i] = await node.status()

        async with anyio.create_task_group() as tg:
            for i, node in enumerate(self.nodes):
                tg.start_soon(_check, i, node)

        return results

    def find_node(self, name: str) -> OllamaNode | None:
        for node in self.nodes:
            if node.config.name == name:
                return node
        return None

    async def find_model(self, model_name: str) -> list[tuple[OllamaNode, ModelInfo]]:
        """Find which nodes have a given model."""
        statuses = await self.status_all()
        results = []
        for status, node in zip(statuses, self.nodes):
            if not status.online:
                continue
            for m in status.models:
                if model_name in m.name:
                    results.append((node, m))
        return results

    async def best_node_for_model(self, model_name: str) -> OllamaNode | None:
        """Pick the node with lowest latency that has the model."""
        candidates = await self.find_model(model_name)
        if not candidates:
            return None
        # Prefer nodes that already have the model loaded/running
        # then sort by latency
        statuses = {n.config.name: await n.ping() for n, _ in candidates}
        candidates.sort(key=lambda x: statuses[x[0].config.name][1])
        return candidates[0][0]
