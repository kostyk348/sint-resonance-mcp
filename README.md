# sint-resonance-mcp

Resonance Detection MCP server — proactively finds cross-domain pattern matches between the current problem and past experiences.

## Tools

| Tool | Description |
|------|-------------|
| `resonance_index(domain, problem, solution, tags)` | Index a problem-solution pair |
| `resonance_find(query, k, domains)` | Find top K resonating past problems |
| `resonance_structural(problem_structure, k)` | Match by structural pattern (not keywords) |
| `resonance_proactive(context_snapshot)` | Auto-suggest resonances from current context |
| `resonance_domains()` | List all domains in index |
| `resonance_stats()` | Index statistics |

## Architecture

```
sint-resonance-mcp/
├── server.py          — FastMCP server
├── config.py          — NAME, VERSION
├── resonance/
│   ├── index.py       — Problem-solution index (persisted to ~/.config/opencode/resonance/)
│   ├── embeddings.py  — TF-IDF + keyword overlap similarity (zero external deps)
│   ├── structural.py  — Structural pattern matching (AST-like fingerprints)
│   └── proactive.py   — Proactive suggestion engine
└── pyproject.toml
```

## Persistence

Data is stored at `~/.config/opencode/resonance/index.json` — survives server restart.
