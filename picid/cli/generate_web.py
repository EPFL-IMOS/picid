"""Generate config_map.json for the Experiment Generator web UI.

Uses config_discovery (same as picid-assemble) so the web generator stays
in sync with the config tree. Run before mkdocs build.
"""

from __future__ import annotations

import json
from pathlib import Path


def main(output_dir: Path | None = None) -> None:
    """Generate config_map.json from configs/ using config_discovery."""
    from picid.cli.config_discovery import (
        list_debug_configs,
        list_experiment_groups,
        list_experiments_for_group_task,
        list_paths_configs,
    )

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = output_dir or (project_root / "docs" / "experiment-generator")

    experiment_keys: list[str] = []
    for group in list_experiment_groups():
        for task_dir in sorted(group.path.iterdir()):
            if not task_dir.is_dir():
                continue
            task_type = task_dir.name
            keys = list_experiments_for_group_task(group.name, task_type)
            experiment_keys.extend(keys)

    # Exclude base configs (not runnable experiments)
    experiment_keys = [k for k in experiment_keys if not k.endswith("/base")]

    config_map = {
        "experiment_keys": sorted(set(experiment_keys)),
        "paths": list_paths_configs() or ["default"],
        "debug": list_debug_configs(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "config_map.json"
    out_path.write_text(json.dumps(config_map, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate config_map.json for Experiment Generator"
    )
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    args = parser.parse_args()
    main(output_dir=args.output)
