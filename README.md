# ollafleet

**Smart proxy making multiple Ollama instances act as one.**

Routes each request to the best upstream node for the requested model:

| priority | rule |
|---|---|
| 1 | nodes that **already have the model loaded** (no cold start) |
| 2 | nodes that have the model **installed** |
| 3 | round-robin across all nodes |

Per-node `/api/tags` and `/api/ps` are cached for 30s. Every response carries an `X-Ollafleet-Node` header so you can trace which node served.

```bash
./ollafleet serve --nodes http://macstudio:11434,http://corsair:11434 \
                  --listen 127.0.0.1:11455
# point any Ollama client at http://127.0.0.1:11455 — speaks plain Ollama

./ollafleet status --nodes http://macstudio:11434,http://corsair:11434
# inventory + loaded models per node, with ● for already-loaded
```

## Where it sits

```
client  →  ollafleet  →  { node A | node B | … }   (each running Ollama)
```

Slots in front of any other proxy: `client → ollafleet → ollafifo → ollasecret → ollama-on-node-X`.

Single file, Python stdlib only.

## License

MIT
