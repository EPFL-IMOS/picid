from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


GOOD = 0
BAD = 1
SKIP = 125
RTOL = 1.0e-7
ATOL = 1.0e-9


def _git(worktree: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(worktree), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _package_runner(worktree: Path) -> Path:
    for relative in ("lmetk/run.py", "picid/run.py"):
        candidate = worktree / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Neither lmetk/run.py nor picid/run.py exists")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_probe(probe_dir: Path) -> dict[str, Any]:
    with (probe_dir / "probe.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _compare_npz(
    reference_path: Path,
    candidate_path: Path,
    *,
    exact: bool,
) -> dict[str, Any] | None:
    if not candidate_path.is_file():
        return {"kind": "missing_artifact", "path": candidate_path.name}
    with np.load(reference_path) as reference, np.load(candidate_path) as candidate:
        reference_keys = set(reference.files)
        candidate_keys = set(candidate.files)
        if reference_keys != candidate_keys:
            return {
                "kind": "tensor_keys",
                "path": reference_path.name,
                "reference_only": sorted(reference_keys - candidate_keys),
                "candidate_only": sorted(candidate_keys - reference_keys),
            }
        for key in sorted(reference_keys):
            expected = reference[key]
            actual = candidate[key]
            if expected.shape != actual.shape or expected.dtype != actual.dtype:
                return {
                    "kind": "tensor_metadata",
                    "path": reference_path.name,
                    "tensor": key,
                    "reference_shape": list(expected.shape),
                    "candidate_shape": list(actual.shape),
                    "reference_dtype": str(expected.dtype),
                    "candidate_dtype": str(actual.dtype),
                }
            equal = (
                np.array_equal(expected, actual, equal_nan=True)
                if exact
                else np.allclose(
                    expected,
                    actual,
                    rtol=RTOL,
                    atol=ATOL,
                    equal_nan=True,
                )
            )
            if not equal:
                difference = np.abs(
                    expected.astype(np.float64) - actual.astype(np.float64)
                )
                return {
                    "kind": "tensor_values",
                    "path": reference_path.name,
                    "tensor": key,
                    "exact_required": exact,
                    "max_abs_difference": float(np.nanmax(difference)),
                }
    return None


def _compare_numbers(
    name: str, reference: list[Any], candidate: list[Any]
) -> dict[str, Any] | None:
    if len(reference) != len(candidate):
        return {
            "kind": "sequence_length",
            "sequence": name,
            "reference": len(reference),
            "candidate": len(candidate),
        }
    expected = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(candidate, dtype=np.float64)
    if not np.allclose(expected, actual, rtol=RTOL, atol=ATOL, equal_nan=True):
        return {
            "kind": "numeric_sequence",
            "sequence": name,
            "reference": reference,
            "candidate": candidate,
            "max_abs_difference": float(np.nanmax(np.abs(expected - actual))),
        }
    return None


def compare_probes(reference_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    reference = _load_probe(reference_dir)
    candidate = _load_probe(candidate_dir)

    reference_batches = sorted(reference_dir.glob("batch_*.npz"))
    candidate_batch_names = {path.name for path in candidate_dir.glob("batch_*.npz")}
    if {path.name for path in reference_batches} != candidate_batch_names:
        return {
            "matches": False,
            "first_divergence": {
                "kind": "batch_files",
                "reference": [path.name for path in reference_batches],
                "candidate": sorted(candidate_batch_names),
            },
        }
    for reference_path in reference_batches:
        divergence = _compare_npz(
            reference_path,
            candidate_dir / reference_path.name,
            exact=True,
        )
        if divergence:
            return {"matches": False, "first_divergence": divergence}

    ordered_artifacts = [
        "initial_parameters.npz",
        *[path.name for path in sorted(reference_dir.glob("train_step_*.npz"))],
        "final_parameters.npz",
    ]
    for filename in ordered_artifacts:
        divergence = _compare_npz(
            reference_dir / filename,
            candidate_dir / filename,
            exact=False,
        )
        if divergence:
            return {"matches": False, "first_divergence": divergence}

    for sequence in ("train_losses", "validation_losses"):
        divergence = _compare_numbers(
            sequence,
            reference.get(sequence, []),
            candidate.get(sequence, []),
        )
        if divergence:
            return {"matches": False, "first_divergence": divergence}

    reference_gradients = [entry["l2"] for entry in reference.get("gradient_norms", [])]
    candidate_gradients = [entry["l2"] for entry in candidate.get("gradient_norms", [])]
    divergence = _compare_numbers(
        "gradient_norms", reference_gradients, candidate_gradients
    )
    if divergence:
        return {"matches": False, "first_divergence": divergence}

    return {"matches": True, "first_divergence": None}


def _probe_metrics(probe_dir: Path) -> dict[str, Any]:
    if not (probe_dir / "probe.json").is_file():
        return {}
    probe = _load_probe(probe_dir)
    return {
        "train_losses": probe.get("train_losses", []),
        "validation_losses": probe.get("validation_losses", []),
        "gradient_norms": [
            entry.get("l2") for entry in probe.get("gradient_norms", [])
        ],
        "batch_digests": {
            batch_index: {
                key: value["sha256"] for key, value in sorted(manifest.items())
            }
            for batch_index, manifest in sorted(probe.get("batches", {}).items())
        },
    }


def run_probe(args: argparse.Namespace) -> int:
    worktree = args.worktree.resolve()
    output_root = args.output_root.resolve()
    results_jsonl = args.results_jsonl.resolve()
    revision = _git(worktree, "rev-parse", "HEAD")
    subject = _git(worktree, "show", "-s", "--format=%s", "HEAD")
    run_id = args.run_id or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = output_root / "runs" / revision / run_id
    probe_dir = run_dir / "probe"
    hydra_dir = run_dir / "hydra"
    run_dir.mkdir(parents=True, exist_ok=False)

    started = time.monotonic()
    timestamp = dt.datetime.now(dt.UTC).isoformat()
    try:
        framework_runner = _package_runner(worktree)
        command = [
            str(args.python.absolute()),
            str(framework_runner),
            "paths=rt_local",
            "experiment=phme20/prognostics/raw/lstm",
            "debug=default",
            "seed=72",
            "trainer.max_epochs=1",
            "trainer.deterministic=true",
            "++trainer.limit_train_batches=3",
            "++trainer.limit_val_batches=2",
            "++trainer.limit_test_batches=0",
            "datamodule.num_workers=0",
            "test=false",
            "num_threads=1",
            "enable_progress_bar=false",
            "cache.use_cache_after_loading=false",
            "cache.use_cache_after_transfroms=false",
            "++cache.use_cache_after_boundary_conditions=false",
            f"hydra.run.dir={hydra_dir}",
            "++callbacks.phme20_probe._target_=phme20_probe_callback.PHME20ProbeCallback",
            f"++callbacks.phme20_probe.output_dir={probe_dir}",
            f"++callbacks.phme20_probe.revision={revision}",
        ]
        environment = os.environ.copy()
        regression_dir = Path(__file__).resolve().parent
        python_path = [str(regression_dir), str(worktree)]
        if environment.get("PYTHONPATH"):
            python_path.append(environment["PYTHONPATH"])
        environment.update(
            {
                "PROJECT_ROOT": str(worktree),
                "PYTHONPATH": os.pathsep.join(python_path),
                "PYTHONHASHSEED": "72",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "",
            }
        )
        with (run_dir / "run.log").open("w", encoding="utf-8") as log:
            log.write("command: " + shlex.join(command) + "\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=worktree,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout,
            )
        returncode = completed.returncode
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        returncode = -1
        with (run_dir / "runner_error.txt").open("w", encoding="utf-8") as handle:
            handle.write(f"{type(exc).__name__}: {exc}\n")

    comparison: dict[str, Any] | None = None
    if returncode != 0 or not (probe_dir / "probe.json").is_file():
        classification = "skip"
        exit_code = SKIP
    elif args.baseline is None:
        classification = "baseline"
        exit_code = GOOD
    else:
        comparison = compare_probes(args.baseline.resolve(), probe_dir)
        classification = "good" if comparison["matches"] else "bad"
        exit_code = GOOD if comparison["matches"] else BAD

    record = {
        "schema_version": 1,
        "timestamp": timestamp,
        "commit": revision,
        "subject": subject,
        "run_id": run_id,
        "role": args.role,
        "classification": classification,
        "process_returncode": returncode,
        "duration_seconds": time.monotonic() - started,
        "artifact_dir": str(run_dir),
        "probe_dir": str(probe_dir),
        "baseline_dir": str(args.baseline.resolve()) if args.baseline else None,
        "comparison": comparison,
        **_probe_metrics(probe_dir),
    }
    _append_jsonl(results_jsonl, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--role", default="candidate")
    parser.add_argument("--timeout", type=int, default=1800)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(run_probe(parse_args()))
