"""Fleet configuration loader."""

from __future__ import annotations

from pathlib import Path

import yaml

from ollafleet.client import NodeConfig

CONFIG_SEARCH_PATHS = [
    Path("fleet.yml"),
    Path.home() / ".config" / "ollafleet" / "fleet.yml",
]


def find_config(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"Config not found: {explicit}")

    for p in CONFIG_SEARCH_PATHS:
        if p.exists():
            return p

    raise FileNotFoundError(
        "No fleet.yml found. Create one in the current directory or "
        "~/.config/ollafleet/fleet.yml. Run 'ollafleet init' to generate one."
    )


def load_config(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data or "nodes" not in data:
        raise ValueError(f"Invalid config: {path} must have a 'nodes' list")
    return data


def parse_nodes(data: dict) -> list[NodeConfig]:
    nodes = []
    for entry in data["nodes"]:
        nodes.append(NodeConfig(
            name=entry["name"],
            host=entry["host"].rstrip("/"),
            tags=entry.get("tags", []),
        ))
    return nodes
