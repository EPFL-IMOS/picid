"""3W datasource loader with predefined fold splits."""

from __future__ import annotations

import logging
from pathlib import Path
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
            logger.warning(
                "Automatic download is not implemented for 3W. "
                "Provide a local 3W dataset directory."
            )

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
        folds_path = self.data_path / self.folds_file
        if not folds_path.exists():
            raise FileNotFoundError(f"3W folds file not found: {folds_path}")

        folds_df = pd.read_csv(folds_path)
        records: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        unit_id = 0

        for row in folds_df.itertuples(index=False):
            if not self.include_ova and bool(row.is_ova):
                continue

            split_name = self._split_name_from_fold(int(row.fold))
            if split_name is None:
                continue

            class_id, instance_name = self._parse_instance_ref(str(row.instancia))
            parquet_path = self._resolve_instance_path(class_id, instance_name)
            frame = self._read_instance_frame(parquet_path)
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

        return records

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
        return self.data_path / "dataset" / str(class_id) / f"{instance_name}.parquet"

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
