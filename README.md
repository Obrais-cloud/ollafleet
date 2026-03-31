# ollafleet

CLI fleet manager for multiple Ollama instances. One terminal, all your nodes.

```
ollafleet status      # health of every node
ollafleet models      # all models across the fleet
ollafleet locate qwen # which nodes have a model?
ollafleet pull llama3.3 # pull to all nodes at once
ollafleet chat "why is the sky blue?" # auto-routes to best node
ollafleet bench       # benchmark tok/s across nodes
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Setup

Create `fleet.yml` (or run `ollafleet init`):

```yaml
nodes:
  - name: corsair
    host: "http://localhost:11434"
    tags: [mothership, compute]

  - name: mac-studio
    host: "http://100.x.x.x:11434"
    tags: [speed-tier]

  - name: alienware
    host: "http://100.x.x.x:11434"
    tags: [specialist, cuda]

default_model: "qwen3.5:27b"
timeout: 15
```

Config is searched in order: `./fleet.yml` → `~/.config/ollafleet/fleet.yml`.

## Commands

| Command | Description |
|---------|-------------|
| `status` | Health, version, latency, model count, running models per node |
| `models` | All models across the fleet with size, params, quantization, loaded state |
| `locate <model>` | Find which nodes have a specific model |
| `pull <model>` | Pull a model to all nodes (or `--node` for one) |
| `chat [prompt]` | Chat with auto-routing to the fastest node that has the model |
| `bench` | Benchmark tok/s across nodes for a given model |
| `init` | Generate a starter fleet.yml |

## Flags

- `-c, --config PATH` — explicit config file path
- `-n, --node NAME` — target a specific node (available on most commands)
- `-m, --model NAME` — specify model (for chat/bench)

## How it works

All nodes are queried in parallel using async HTTP. Chat auto-routes to the node with the lowest latency that has the requested model. Pull runs concurrently across all nodes.

## License

MIT
