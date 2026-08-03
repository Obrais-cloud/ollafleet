"""ollafleet CLI — manage your Ollama fleet from one terminal."""

from __future__ import annotations

import sys
from pathlib import Path

import anyio
import click
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.panel import Panel

from ollafleet.client import Fleet, NodeConfig
from ollafleet.config import find_config, load_config, parse_nodes

console = Console()


def _load_fleet(config_path: str | None) -> tuple[Fleet, dict]:
    try:
        path = find_config(config_path)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    data = load_config(path)
    nodes = parse_nodes(data)
    timeout = data.get("timeout", 15)
    return Fleet(nodes, timeout), data


@click.group()
@click.option("--config", "-c", default=None, help="Path to fleet.yml")
@click.pass_context
def cli(ctx, config):
    """ollafleet — manage your Ollama fleet from one terminal."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@cli.command()
@click.pass_context
def status(ctx):
    """Show health and status of all fleet nodes."""
    fleet, _ = _load_fleet(ctx.obj["config_path"])

    async def _run():
        try:
            statuses = await fleet.status_all()
        finally:
            await fleet.close()

        table = Table(title="Fleet Status", show_lines=True)
        table.add_column("Node", style="bold cyan")
        table.add_column("Status")
        table.add_column("Version")
        table.add_column("Latency")
        table.add_column("Models")
        table.add_column("Running")
        table.add_column("Tags", style="dim")

        for s in statuses:
            if s.online:
                status_str = "[green]● online[/green]"
                version = s.version or "?"
                latency = f"{s.latency_ms:.0f}ms"
                model_count = str(len(s.models))
                running_names = [r.get("name", "?") for r in s.running]
                running_str = ", ".join(running_names) if running_names else "[dim]none[/dim]"
            else:
                status_str = "[red]● offline[/red]"
                version = "-"
                latency = "-"
                model_count = "-"
                running_str = f"[red]{s.error or 'unreachable'}[/red]"

            tags = ", ".join(s.tags) if s.tags else ""
            table.add_row(s.name, status_str, version, latency, model_count, running_str, tags)

        console.print(table)

    anyio.run(_run)


@cli.command()
@click.option("--node", "-n", default=None, help="Filter by node name")
@click.pass_context
def models(ctx, node):
    """List all models across the fleet."""
    fleet, _ = _load_fleet(ctx.obj["config_path"])

    async def _run():
        try:
            statuses = await fleet.status_all()
        finally:
            await fleet.close()

        table = Table(title="Fleet Models", show_lines=True)
        table.add_column("Model", style="bold")
        table.add_column("Node", style="cyan")
        table.add_column("Size")
        table.add_column("Params")
        table.add_column("Quant")
        table.add_column("Loaded", justify="center")

        for s in statuses:
            if not s.online:
                continue
            if node and s.name != node:
                continue
            running_names = {r.get("name", "") for r in s.running}
            for m in sorted(s.models, key=lambda x: x.name):
                loaded = "[green]●[/green]" if m.name in running_names else "[dim]○[/dim]"
                table.add_row(
                    m.name,
                    s.name,
                    f"{m.size_gb:.1f} GB",
                    m.parameter_size,
                    m.quantization,
                    loaded,
                )

        console.print(table)

    anyio.run(_run)


@cli.command()
@click.argument("model")
@click.option("--node", "-n", default=None, help="Pull to specific node (default: all)")
@click.pass_context
def pull(ctx, model, node):
    """Pull a model to one or all fleet nodes."""
    fleet, _ = _load_fleet(ctx.obj["config_path"])

    async def _pull_one(ollama_node):
        console.print(f"[cyan]{ollama_node.config.name}[/cyan]: pulling [bold]{model}[/bold]...")
        try:
            last_status = ""
            async for progress in ollama_node.pull_model(model):
                status = progress.get("status", "")
                if status != last_status:
                    console.print(f"  [dim]{ollama_node.config.name}:[/dim] {status}")
                    last_status = status
            console.print(f"[green]✓[/green] {ollama_node.config.name}: {model} pulled successfully")
        except Exception as e:
            console.print(f"[red]✗[/red] {ollama_node.config.name}: {e}")

    async def _run():
        try:
            if node:
                target = fleet.find_node(node)
                if not target:
                    console.print(f"[red]Node '{node}' not found in fleet config[/red]")
                    return
                await _pull_one(target)
            else:
                async with anyio.create_task_group() as tg:
                    for n in fleet.nodes:
                        tg.start_soon(_pull_one, n)
        finally:
            await fleet.close()

    anyio.run(_run)


@cli.command()
@click.argument("prompt", nargs=-1, required=False)
@click.option("--model", "-m", default=None, help="Model to use")
@click.option("--node", "-n", default=None, help="Force specific node")
@click.pass_context
def chat(ctx, prompt, model, node):
    """Chat with a model. Auto-routes to the best available node."""
    fleet, data = _load_fleet(ctx.obj["config_path"])
    model = model or data.get("default_model", "llama3.3:latest")
    prompt_text = " ".join(prompt) if prompt else None

    async def _run():
        try:
            if node:
                target = fleet.find_node(node)
                if not target:
                    console.print(f"[red]Node '{node}' not found[/red]")
                    return
            else:
                target = await fleet.best_node_for_model(model)
                if not target:
                    console.print(f"[red]No node has model '{model}'. Run: ollafleet pull {model}[/red]")
                    return

            console.print(
                f"[dim]→ {target.config.name} ({target.config.host}) · {model}[/dim]"
            )

            messages: list[dict] = []

            if prompt_text:
                # One-shot mode
                messages.append({"role": "user", "content": prompt_text})
                async for chunk in target.chat_stream(model, messages):
                    console.print(chunk, end="", highlight=False)
                console.print()
            else:
                # Interactive mode
                console.print("[dim]Interactive chat. Type 'quit' or Ctrl+C to exit.[/dim]")
                while True:
                    try:
                        user_input = console.input("[bold green]you>[/bold green] ")
                    except (EOFError, KeyboardInterrupt):
                        console.print("\n[dim]bye[/dim]")
                        break
                    if user_input.strip().lower() in ("quit", "exit", "q"):
                        break
                    messages.append({"role": "user", "content": user_input})
                    console.print("[bold blue]ai>[/bold blue] ", end="")
                    full_response = ""
                    async for chunk in target.chat_stream(model, messages):
                        console.print(chunk, end="", highlight=False)
                        full_response += chunk
                    console.print()
                    messages.append({"role": "assistant", "content": full_response})
        finally:
            await fleet.close()

    anyio.run(_run)


@cli.command()
@click.option("--model", "-m", default=None, help="Model to benchmark")
@click.option("--node", "-n", default=None, help="Bench specific node only")
@click.option("--prompt", "-p", default="Explain quantum entanglement in one paragraph.", help="Prompt to use")
@click.pass_context
def bench(ctx, model, node, prompt):
    """Benchmark inference speed across fleet nodes."""
    fleet, data = _load_fleet(ctx.obj["config_path"])
    model = model or data.get("default_model", "llama3.3:latest")

    async def _bench_node(ollama_node, results: list):
        name = ollama_node.config.name
        try:
            models = await ollama_node.list_models()
            model_names = [m.name for m in models]
            if not any(model in m for m in model_names):
                results.append((name, None, None, f"model '{model}' not found"))
                return
            console.print(f"[dim]Benchmarking {name}...[/dim]")
            response, tok_s = await ollama_node.generate_once(model, prompt)
            results.append((name, tok_s, len(response), None))
        except Exception as e:
            results.append((name, None, None, str(e)))

    async def _run():
        try:
            results: list = []
            targets = fleet.nodes
            if node:
                target = fleet.find_node(node)
                if not target:
                    console.print(f"[red]Node '{node}' not found[/red]")
                    return
                targets = [target]

            # Run benchmarks sequentially to not interfere with each other
            for n in targets:
                await _bench_node(n, results)

            table = Table(title=f"Benchmark: {model}")
            table.add_column("Node", style="bold cyan")
            table.add_column("tok/s", justify="right")
            table.add_column("Response len", justify="right")
            table.add_column("Status")

            for name, tok_s, resp_len, err in results:
                if err:
                    table.add_row(name, "-", "-", f"[red]{err}[/red]")
                else:
                    table.add_row(
                        name,
                        f"[bold green]{tok_s:.1f}[/bold green]",
                        str(resp_len),
                        "[green]ok[/green]",
                    )
            console.print(table)
        finally:
            await fleet.close()

    anyio.run(_run)


@cli.command()
@click.argument("model")
@click.pass_context
def locate(ctx, model):
    """Find which nodes have a specific model."""
    fleet, _ = _load_fleet(ctx.obj["config_path"])

    async def _run():
        try:
            results = await fleet.find_model(model)
            if not results:
                console.print(f"[yellow]'{model}' not found on any node[/yellow]")
                return
            for node_obj, model_info in results:
                console.print(
                    f"[green]●[/green] [cyan]{node_obj.config.name}[/cyan] — "
                    f"{model_info.name} ({model_info.size_gb:.1f} GB, {model_info.quantization})"
                )
        finally:
            await fleet.close()

    anyio.run(_run)


@cli.command()
@click.argument("path", default=".", required=False)
def init(path):
    """Generate a fleet.yml config file."""
    target = Path(path) / "fleet.yml"
    if target.exists():
        console.print(f"[yellow]{target} already exists[/yellow]")
        return

    template = """\
# ollafleet configuration
nodes:
  - name: local
    host: "http://localhost:11434"
    tags: [default]

  # Add more nodes:
  # - name: remote-gpu
  #   host: "http://192.168.1.100:11434"
  #   tags: [cuda, fast]

default_model: "llama3.3:latest"
timeout: 15
"""
    target.write_text(template)
    console.print(f"[green]✓[/green] Created {target}")


if __name__ == "__main__":
    cli()
