"""
Export an evaluation run to a Unity-friendly playback JSON file.

The input is a run directory produced by evaluate_checkpoint.py. The output keeps
the raw snapshots, metrics, and summary together so Unity only needs to load one
file from StreamingAssets.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from marl_economy.config.constants import (
    UNITY_EXPORT_FILENAME,
    UNITY_REPLAY_SECONDS_PER_WEEK,
    UNITY_SCENE_SCALE,
)


UNITY_SCHEMA_VERSION = 1


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_csv_value(value: str) -> Any:
    if value is None:
        return None
    if value == "":
        return ""
    try:
        if "." not in value and "e" not in value.lower():
            return int(value)
        return float(value)
    except ValueError:
        return value


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [
            {key: _parse_csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(f)
        ]


def export_unity_playback(run_dir: str | Path, output_path: str | Path | None = None) -> str:
    """
    Build a single JSON file for Unity playback.

    Args:
        run_dir: Evaluation directory containing snapshots.json, metrics_step.csv,
            summary.json, and config.json.
        output_path: Optional destination path. Defaults to
            <run_dir>/unity_playback.json.

    Returns:
        Absolute output path.
    """
    run_dir = Path(run_dir).resolve()
    snapshots_payload = _load_json(run_dir / "snapshots.json", {})
    snapshots = snapshots_payload.get("snapshots", [])
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {run_dir / 'snapshots.json'}")

    payload = {
        "schema_version": UNITY_SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(run_dir),
        "replay_seconds_per_week": UNITY_REPLAY_SECONDS_PER_WEEK,
        "scene_scale": UNITY_SCENE_SCALE,
        "mode": snapshots_payload.get("mode", "unknown"),
        "checkpoint_path": snapshots_payload.get("checkpoint_path"),
        "regime": snapshots_payload.get("regime"),
        "stage": snapshots_payload.get("stage") or 0,
        "seed": snapshots_payload.get("seed"),
        "summary": _load_json(run_dir / "summary.json", {}),
        "config": _load_json(run_dir / "config.json", {}),
        "metrics": _load_metrics(run_dir / "metrics_step.csv"),
        "snapshots": snapshots,
    }

    if output_path is None:
        output_path = run_dir / UNITY_EXPORT_FILENAME
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"Unity playback exported: {output_path}")
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an evaluation run for Unity playback")
    parser.add_argument("run_dir", help="Evaluation run directory")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()
    export_unity_playback(args.run_dir, args.output)


if __name__ == "__main__":
    main()
