"""Run serialization: save and load cortex run outputs for dashboard generation."""

import json
import pickle
from datetime import UTC, datetime
from pathlib import Path

from step.cortex.topology import CortexResult
from step.probes.diagnostics import CortexDiagnostics
from step.probes.timeline import Timeline

RUNS_DIR = Path("experiments/runs")


def save_run(
    *,
    name: str,
    timelines: dict[str, Timeline],
    diagnostics: dict[str, CortexDiagnostics],
    result: CortexResult,
    region_configs: dict[str, dict],
    meta_extra: dict | None = None,
) -> Path:
    """Save run outputs to experiments/runs/{name}--{timestamp}/.

    Returns the run directory path.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"{name}--{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Data pickle: everything the dashboard needs
    data = {
        "timelines": timelines,
        "diagnostics": diagnostics,
        "result": result,
        "region_configs": region_configs,
    }
    with open(run_dir / "data.pkl", "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # JSON metadata sidecar: human-readable, used by index page
    meta = {
        "name": name,
        "created": datetime.now(UTC).isoformat(),
        "regions": list(region_configs.keys()),
        "region_configs": region_configs,
        "elapsed_seconds": result.elapsed_seconds,
    }
    if meta_extra:
        meta.update(meta_extra)

    # Extract summary stats for index display
    for region_name, metrics in result.per_region.items():
        rep = metrics.representation
        if rep:
            meta.setdefault("summary", {})[region_name] = {
                "ctx_disc": rep.get("context_discrimination", 0),
                "selectivity": rep.get("column_selectivity_mean", 0),
            }

    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved run to {run_dir}")
    return run_dir


def load_run(run_dir: str | Path) -> dict:
    """Load run data from a directory.

    Returns dict with keys: timelines, diagnostics, result, region_configs.
    """
    run_dir = Path(run_dir)
    with open(run_dir / "data.pkl", "rb") as f:
        return pickle.load(f)


def load_meta(run_dir: str | Path) -> dict:
    """Load just the JSON metadata (fast, no pickle)."""
    run_dir = Path(run_dir)
    with open(run_dir / "meta.json") as f:
        return json.load(f)


def list_runs(runs_dir: Path = RUNS_DIR) -> list[Path]:
    """List all run directories, newest first."""
    if not runs_dir.exists():
        return []
    dirs = [
        d for d in runs_dir.iterdir()
        if d.is_dir() and (d / "meta.json").exists()
    ]
    dirs.sort(key=lambda d: d.name, reverse=True)
    return dirs


def auto_name(
    *,
    hierarchy: bool = False,
    char_level: bool = False,
    n_tokens: int = 0,
    buffer_depth: int = 1,
    burst_gate: bool = False,
    apical: bool = False,
) -> str:
    """Derive a run name from parameters."""
    parts = []
    if hierarchy:
        parts.append("hierarchy")
    else:
        parts.append("single")
    if char_level:
        parts.append("char")
    if buffer_depth > 1:
        parts.append(f"buf{buffer_depth}")
    if burst_gate:
        parts.append("burst")
    if apical:
        parts.append("apical")
    if n_tokens:
        if n_tokens >= 1000:
            parts.append(f"{n_tokens // 1000}k")
        else:
            parts.append(f"{n_tokens}")
    return "-".join(parts)


def auto_tags(
    *,
    hierarchy: bool = False,
    char_level: bool = False,
    buffer_depth: int = 1,
    burst_gate: bool = False,
    apical: bool = False,
) -> list[str]:
    """Derive tags from parameters."""
    tags = []
    if hierarchy:
        tags.append("hierarchy")
    if char_level:
        tags.append("char-level")
    if buffer_depth > 1:
        tags.append(f"buffer-{buffer_depth}")
    if burst_gate:
        tags.append("burst-gate")
    if apical:
        tags.append("apical")
    return tags
