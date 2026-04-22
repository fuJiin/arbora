# Examples

End-to-end applications built on Arbora. Each subdirectory is a self-contained
package — invoke via `python -m` so the cross-file imports resolve:

```bash
uv sync --extra minigrid
uv run python -m examples.minigrid.train --episodes 100
uv run python -m examples.minigrid.benchmark --episodes 1000

uv sync --extra arc
uv run python -m examples.arc.train --keyboard-only --episodes 5

uv run python -m examples.chat.train
```

## Stability

> **Heads up:** these examples live in the core `arbora` repo today for
> convenience during alpha development, but most will be moved out into their
> own repositories once Arbora is packaged for external use. Treat anything
> under `examples/` as research scaffolding: APIs, CLI flags, and file layout
> can change or disappear between commits.

If you're building on top of Arbora, import from `arbora.*` (the public
package), not from `examples.*`.

## What's here

- **`arc/`** — ARC-AGI-3 spatial reasoning with a transthalamic hierarchy
  (V1 → pulvinar → V2 → BG → M1). Requires the `arc` extra.
- **`chat/`** — Character-level text learning with a sensory–motor hierarchy
  (T1 → T2 → T3 → PFC → M2 → M1).
- **`minigrid/`** — Grid navigation with MiniGrid (T1 → BG → M1). Requires
  the `minigrid` extra.
