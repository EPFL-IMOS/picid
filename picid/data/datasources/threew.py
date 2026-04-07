"""3W datasource loader with predefined fold splits."""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from picid.data.datasources.base.predefined_split_loader import (
    PredefinedSplitLoaderBase,
)
from picid.data.datasources.utils import convert_outer_list_to_inner

logger = logging.getLogger(__name__)


class ThreeWLoader(PredefinedSplitLoaderBase):
    """
    Load Petrobras 3W instances with predefined fold assignments.

    Parameters
    ----------
    data_dir : str
        Root directory containing the local 3W ``dataset`` folder.
    data_name : str
        Datasource name exposed to the pipeline.
    task_mode : str
        Task mode associated with this datasource.
    folds_file : str, optional
        Relative path to the folds CSV file from ``data_dir``.
    validation_fold : int, optional
        Fold id to use as validation split.
    test_fold : int, optional
        Fold id to use as test split.
    include_ova : bool, optional
        Whether to include one-vs-all rows marked in the folds file.
    include_simulated_train : bool, optional
        Whether simulated rows (fold ``-1``) are assigned to train.
    export_event_class : bool, optional
        Whether to export per-timestep event class arrays.
    download : bool, optional
        Download flag kept for API compatibility.
    **kwargs : dict[str, Any]
        Forwarded datasource compatibility arguments.

    Notes
    -----
    The loader follows the dataset fold file to create train/val/test splits and
    emits a binary anomaly target (class ``0`` is healthy, all others anomalous).
    Original event classes are stored in metadata and can optionally be exported
    as ``event_class`` for downstream multi-class experiments.
    """

    DEFAULT_FEATURE_COLUMNS = (
        "P-PDG",
        "P-TPT",
        "T-TPT",
        "P-MON-CKP",
        "T-JUS-CKP",
        "P-JUS-CKGL",
        "T-JUS-CKGL",
        "QGL",
    )
    _REPO_URL = "https://github.com/petrobras/3W.git"

    def __init__(
        self,
        *,
        data_dir: str,
        data_name: str,
        task_mode: str,
        folds_file: str = "dataset/folds/folds_clf_02.csv",
        validation_fold: int = 0,
        test_fold: int = 1,
        include_ova: bool = False,
        include_simulated_train: bool = True,
        export_event_class: bool = False,
        download: bool = False,
        **kwargs,
    ):
        """
        Initialize the 3W predefined-split loader.

        Parameters
        ----------
        data_dir : str
            Root directory containing the local 3W ``dataset`` folder.
        data_name : str
            Datasource name exposed to the pipeline.
        task_mode : str
            Task mode associated with this datasource.
        folds_file : str, optional
            Relative path to the folds CSV file from ``data_dir``.
        validation_fold : int, optional
            Fold id to use as validation split.
        test_fold : int, optional
            Fold id to use as test split.
        include_ova : bool, optional
            Whether to include one-vs-all rows marked in the folds file.
        include_simulated_train : bool, optional
            Whether simulated rows (fold ``-1``) are assigned to train.
        export_event_class : bool, optional
            Whether to export per-timestep event class arrays.
        download : bool, optional
            Download flag kept for API compatibility.
        **kwargs : dict[str, Any]
            Forwarded datasource compatibility arguments.
        """
        self.data_path = Path(data_dir)
        self.folds_file = folds_file
        self.validation_fold = validation_fold
        self.test_fold = test_fold
        self.include_ova = include_ova
        self.include_simulated_train = include_simulated_train
        self.export_event_class = export_event_class
        self.download = download
        self._resolved_dataset_roots: tuple[Path, ...] = ()
        super().__init__(
            data_name=data_name,
            task_mode=task_mode,
            **kwargs,
        )

    def _load_data(self) -> dict[str, Any]:
        """
        Load split payloads and derive split-wise metadata.

        Returns
        -------
        dict[str, Any]
            Dictionary keyed by tensor name, each containing train/val/test
            lists with one entry per instance.
        """
        if self.download:
            logger.info("3W auto-download is enabled when local folds are missing.")

        records_by_split = self.read_data()
        train = convert_outer_list_to_inner(records_by_split["train"])
        val = convert_outer_list_to_inner(records_by_split["val"])
        test = convert_outer_list_to_inner(records_by_split["test"])

        keys = {"features", "target", "unit_id", "metadata"}
        if self.export_event_class:
            keys.add("event_class")

        out_dict = {}
        for key in keys:
            out_dict[key] = {
                "train": train.get(key, []),
                "val": val.get(key, []),
                "test": test.get(key, []),
            }

        # Keep loader metadata in a split-wise shape for pipeline compatibility.
        self.meta_data.update(
            {
                "unit_ids": {
                    "train": [
                        int(np.asarray(v).reshape(-1)[0])
                        for v in out_dict["unit_id"]["train"]
                    ],
                    "val": [
                        int(np.asarray(v).reshape(-1)[0])
                        for v in out_dict["unit_id"]["val"]
                    ],
                    "test": [
                        int(np.asarray(v).reshape(-1)[0])
                        for v in out_dict["unit_id"]["test"]
                    ],
                },
                "unit_names": {
                    "train": [m["unit_name"] for m in out_dict["metadata"]["train"]],
                    "val": [m["unit_name"] for m in out_dict["metadata"]["val"]],
                    "test": [m["unit_name"] for m in out_dict["metadata"]["test"]],
                },
                "class_labels": {
                    "train": [m["class_label"] for m in out_dict["metadata"]["train"]],
                    "val": [m["class_label"] for m in out_dict["metadata"]["val"]],
                    "test": [m["class_label"] for m in out_dict["metadata"]["test"]],
                },
            }
        )

        return out_dict

    def read_data(self) -> dict[str, list[dict[str, Any]]]:
        """
        Read 3W folds and return split-wise payload records.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Mapping from split names (``train``, ``val``, ``test``) to per-unit
            payload records containing features, targets, ids, and metadata.
        """
        folds_path = self._resolve_folds_path()
        self._resolved_dataset_roots = tuple(self._candidate_dataset_roots(folds_path))

        folds_df = pd.read_csv(folds_path)
        records: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        unit_id = 0
        attempted_rows = 0
        skipped_missing_rows = 0

        for row in folds_df.itertuples(index=False):
            if not self.include_ova and bool(row.is_ova):
                continue

            split_name = self._split_name_from_fold(int(row.fold))
            if split_name is None:
                continue
            attempted_rows += 1

            class_id, instance_name = self._parse_instance_ref(str(row.instancia))
            parquet_path = self._resolve_instance_path(class_id, instance_name)
            try:
                frame = self._read_instance_frame(parquet_path)
            except FileNotFoundError:
                skipped_missing_rows += 1
                continue
            records[split_name].append(
                self._build_record(
                    frame=frame,
                    class_id=class_id,
                    unit_id=unit_id,
                    unit_name=f"{class_id}/{instance_name}",
                    source_path=parquet_path,
                )
            )
            unit_id += 1

        loaded_rows = unit_id
        if attempted_rows > 0 and loaded_rows == 0:
            raise FileNotFoundError(
                "3W folds were found, but no instance parquet files could be loaded. "
                f"Attempted rows={attempted_rows}, skipped_missing_rows={skipped_missing_rows}. "
                "Check that the local 3W dataset version matches the folds file."
            )
        if skipped_missing_rows > 0:
            logger.warning(
                "3W loader skipped %s rows with missing parquet files (%s loaded).",
                skipped_missing_rows,
                loaded_rows,
            )
        return records

    def _resolve_folds_path(self) -> Path:
        """
        Resolve the folds CSV path, optionally attempting automatic dataset fetch.

        Returns
        -------
        pathlib.Path
            Existing path to the resolved 3W folds file.
        """
        existing = self._find_existing_folds_path()
        if existing is not None:
            return existing

        if self.download:
            self._attempt_auto_download()
            existing = self._find_existing_folds_path()
            if existing is not None:
                return existing

        attempted = [str(p) for p in self._candidate_folds_paths()]
        attempted_text = "\n".join(f"  - {p}" for p in attempted)
        raise FileNotFoundError(
            "3W folds file not found. Checked paths:\n"
            f"{attempted_text}\n"
            "Pass --data-dir pointing to a local 3W dataset root, or keep "
            "download=True with network access."
        )

    def _candidate_folds_paths(self) -> list[Path]:
        """
        Enumerate likely folds-file locations for local and downloaded layouts.

        Returns
        -------
        list[pathlib.Path]
            Ordered list of candidate folds CSV paths to probe.
        """
        primary_data_path = self.data_path
        main_repo_data_path = self._worktree_main_repo_data_path()
        candidates = [
            primary_data_path / self.folds_file,
            primary_data_path / "threew_dataset" / "folds" / "folds_clf_02.csv",
            primary_data_path / "dataset" / "folds" / "folds_clf_02.csv",
            primary_data_path / "folds" / "folds_clf_02.csv",
            primary_data_path / "threew" / "dataset" / "folds" / "folds_clf_02.csv",
            primary_data_path / "3w" / "dataset" / "folds" / "folds_clf_02.csv",
        ]
        if main_repo_data_path is not None and main_repo_data_path != primary_data_path:
            candidates.extend(
                [
                    main_repo_data_path / self.folds_file,
                    main_repo_data_path
                    / "threew_dataset"
                    / "folds"
                    / "folds_clf_02.csv",
                    main_repo_data_path / "dataset" / "folds" / "folds_clf_02.csv",
                    main_repo_data_path / "folds" / "folds_clf_02.csv",
                    main_repo_data_path
                    / "threew"
                    / "dataset"
                    / "folds"
                    / "folds_clf_02.csv",
                    main_repo_data_path
                    / "3w"
                    / "dataset"
                    / "folds"
                    / "folds_clf_02.csv",
                ]
            )
        deduped: list[Path] = []
        seen: set[Path] = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            deduped.append(path)
        return deduped

    def _find_existing_folds_path(self) -> Path | None:
        """
        Return the first existing folds path from known candidate locations.

        Returns
        -------
        pathlib.Path | None
            First existing folds path, or ``None`` when none are available.
        """
        for candidate in self._candidate_folds_paths():
            if candidate.exists():
                return candidate
        return None

    def _attempt_auto_download(self) -> None:
        """Best-effort download of the public Petrobras 3W repository."""
        git_bin = shutil.which("git")
        if git_bin is None:
            logger.warning(
                "Automatic 3W download requested but 'git' is unavailable on PATH."
            )
            return

        download_root = self._worktree_main_repo_data_path() or self.data_path
        # Prefer a dedicated subdir to avoid polluting root data_path.
        clone_target = download_root / "threew"
        dataset_marker = clone_target / "dataset" / "folds" / "folds_clf_02.csv"
        if dataset_marker.exists():
            return
        if clone_target.exists() and any(clone_target.iterdir()):
            # Retry from a clean directory if a previous clone was interrupted.
            shutil.rmtree(clone_target, ignore_errors=True)

        clone_target.parent.mkdir(parents=True, exist_ok=True)
        cmd = [git_bin, "clone", "--depth", "1", self._REPO_URL, str(clone_target)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            logger.warning(
                "3W auto-download failed (exit=%s): %s",
                proc.returncode,
                (proc.stderr or proc.stdout).strip(),
            )

    def _candidate_dataset_roots(self, folds_path: Path) -> list[Path]:
        """
        Enumerate dataset roots that may contain class parquet directories.

        Parameters
        ----------
        folds_path : pathlib.Path
            Resolved folds-file path used to infer nearby dataset roots.

        Returns
        -------
        list[pathlib.Path]
            Existing dataset roots to search for class parquet files.
        """
        main_repo_data_path = self._worktree_main_repo_data_path()
        candidates = [
            self.data_path / "dataset",
            self.data_path / "threew_dataset" / "dataset",
            self.data_path / "threew" / "dataset",
            self.data_path / "3w" / "dataset",
        ]
        if main_repo_data_path is not None and main_repo_data_path != self.data_path:
            candidates.extend(
                [
                    main_repo_data_path / "dataset",
                    main_repo_data_path / "threew_dataset" / "dataset",
                    main_repo_data_path / "threew" / "dataset",
                    main_repo_data_path / "3w" / "dataset",
                ]
            )
        parent = folds_path.parent
        if parent.name == "folds":
            candidates.append(parent.parent)
            candidates.append(parent.parent / "dataset")

        deduped: list[Path] = []
        seen: set[Path] = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            if path.exists():
                deduped.append(path)
        return deduped

    def _worktree_main_repo_data_path(self) -> Path | None:
        """
        Resolve main-repository datasets path when running from a git worktree.

        Returns
        -------
        pathlib.Path | None
            Main-repository datasets path, or ``None`` outside worktree layouts.
        """
        parts = self.data_path.resolve().parts
        if ".worktrees" not in parts:
            return None
        idx = parts.index(".worktrees")
        if idx < 1:
            return None
        main_repo_root = Path(*parts[:idx])
        return (main_repo_root / "datasets").resolve()

    def _split_name_from_fold(self, fold: int) -> str | None:
        """
        Map a fold id into a split name.

        Parameters
        ----------
        fold : int
            Fold id from the 3W folds CSV.

        Returns
        -------
        str | None
            Split name or ``None`` when the row should be skipped.
        """
        if fold == self.validation_fold:
            return "val"
        if fold == self.test_fold:
            return "test"
        if fold == -1:
            return "train" if self.include_simulated_train else None
        return "train" if fold >= 0 else None

    def _parse_instance_ref(self, instance_ref: str) -> tuple[int, str]:
        """
        Parse class id and instance stem from a folds file entry.

        Parameters
        ----------
        instance_ref : str
            Folds entry in the form ``"<class_id>/<filename>.csv"``.

        Returns
        -------
        tuple[int, str]
            Parsed class id and parquet filename stem.
        """
        # Example fold format: "3/WELL-00014_20170920060228.csv".
        class_part, filename = instance_ref.split("/", maxsplit=1)
        class_id = int(class_part)
        instance_name = Path(filename).stem
        return class_id, instance_name

    def _resolve_instance_path(self, class_id: int, instance_name: str) -> Path:
        """
        Resolve the expected local parquet path for a 3W instance.

        Parameters
        ----------
        class_id : int
            Event class directory id.
        instance_name : str
            Instance stem without extension.

        Returns
        -------
        pathlib.Path
            Local parquet path inside the 3W dataset directory.
        """
        candidate_roots = (
            list(self._resolved_dataset_roots)
            if self._resolved_dataset_roots
            else [self.data_path / "dataset"]
        )
        fallback = candidate_roots[0] / str(class_id) / f"{instance_name}.parquet"
        for root in candidate_roots:
            candidate = root / str(class_id) / f"{instance_name}.parquet"
            if candidate.exists():
                return candidate
            class_dir = root / str(class_id)
            well_prefix = instance_name.split("_", maxsplit=1)[0]
            fallback_matches = sorted(class_dir.glob(f"{well_prefix}_*.parquet"))
            if fallback_matches:
                return fallback_matches[0]
        return fallback

    def _read_instance_frame(self, parquet_path: Path) -> pd.DataFrame:
        """
        Load one 3W parquet instance from disk.

        Parameters
        ----------
        parquet_path : pathlib.Path
            Local path to the parquet instance file.

        Returns
        -------
        pandas.DataFrame
            Raw instance data for one 3W segment.
        """
        if not parquet_path.exists():
            raise FileNotFoundError(f"3W instance file not found: {parquet_path}")
        return pd.read_parquet(parquet_path)

    def _build_record(
        self,
        *,
        frame: pd.DataFrame,
        class_id: int,
        unit_id: int,
        unit_name: str,
        source_path: Path,
    ) -> dict[str, Any]:
        """
        Build one pipeline record from a single raw 3W instance frame.

        Parameters
        ----------
        frame : pandas.DataFrame
            Raw loaded frame for one instance.
        class_id : int
            Original 3W class label.
        unit_id : int
            Sequential unit identifier in this loading run.
        unit_name : str
            Stable unit name exposed in metadata.
        source_path : pathlib.Path
            Source parquet path for provenance.

        Returns
        -------
        dict[str, Any]
            Unit payload with features, binary target, id, and metadata.
        """
        feature_columns = [
            c for c in self.DEFAULT_FEATURE_COLUMNS if c in frame.columns
        ]
        if not feature_columns:
            excluded = {"timestamp", "class", "state"}
            feature_columns = [
                c
                for c in frame.columns
                if c not in excluded and np.issubdtype(frame[c].dtype, np.number)
            ]
        if not feature_columns:
            raise ValueError(f"No usable numeric features found for {source_path}")

        n_rows = len(frame)
        features = frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
        binary_target = np.full(
            (n_rows, 1), 0.0 if class_id == 0 else 1.0, dtype=np.float32
        )
        event_class = np.full((n_rows, 1), class_id, dtype=np.int64)

        record: dict[str, Any] = {
            "features": features,
            "target": binary_target,
            "unit_id": np.array([unit_id], dtype=np.int64),
            "metadata": {
                "unit_name": unit_name,
                "unit_id": unit_id,
                "class_label": class_id,
                "n_cycles": n_rows,
                "feature_columns": feature_columns,
                "source_path": str(source_path),
            },
        }
        if self.export_event_class:
            record["event_class"] = event_class
        return record
