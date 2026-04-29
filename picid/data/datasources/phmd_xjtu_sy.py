"""
Provide datasource helpers for phmd xjtu sy.
"""

from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any

import awkward as ak
import numpy as np
import pandas as pd

from picid.data.datasources.base.phmd_loader import PHMDMultiSourceLoader

logger = logging.getLogger(__name__)

UNIT_NAMES_TO_ID = {
    "1_1": (1, 1),
    "1_2": (1, 2),
    "1_3": (1, 3),
    "1_4": (1, 4),
    "1_5": (1, 5),
    "2_1": (2, 1),
    "2_2": (2, 2),
    "2_3": (2, 3),
    "2_4": (2, 4),
    "2_5": (2, 5),
    "3_1": (3, 1),
    "3_2": (3, 2),
    "3_3": (3, 3),
    "3_4": (3, 4),
    "3_5": (3, 5),
}
SAMPLE_RANGE_SIZE = 32768
FAULT_TARGET_COLS = ["Outer race", "Cage", "Inner race"]

# Custom evaluation protocols used in the paper. The PHMD payload still serves as
# the transport/source dataset, but these tables redefine which fully observed
# bearings belong to train/validation/test for reporting.
XJTU_SY_SPLIT_ASSIGNMENTS_BY_MODE = {
    "phmd_split": {
        "train": ["1_3", "1_4", "2_1", "2_4", "2_5", "3_1", "3_2", "3_3"],
        "val": ["1_1", "1_2", "3_5"],
        "test": ["1_5", "2_2", "2_3", "3_4"],
    },
    "in_domain": {
        "fold_1": {
            "train": ["1_3", "1_4", "1_5", "2_3", "2_4", "2_5", "3_3", "3_4", "3_5"],
            "val": ["1_1", "2_1", "3_1"],
            "test": ["1_2", "2_2", "3_2"],
        },
        "fold_2": {
            "train": ["1_1", "1_4", "1_5", "2_1", "2_4", "2_5", "3_1", "3_4", "3_5"],
            "val": ["1_2", "2_2", "3_2"],
            "test": ["1_3", "2_3", "3_3"],
        },
        "fold_3": {
            "train": ["1_1", "1_2", "1_5", "2_1", "2_2", "2_5", "3_1", "3_2", "3_5"],
            "val": ["1_3", "2_3", "3_3"],
            "test": ["1_4", "2_4", "3_4"],
        },
        "fold_4": {
            "train": ["1_1", "1_2", "1_3", "2_1", "2_2", "2_3", "3_1", "3_2", "3_3"],
            "val": ["1_4", "2_4", "3_4"],
            "test": ["1_5", "2_5", "3_5"],
        },
        "fold_5": {
            "train": ["1_2", "1_3", "1_4", "2_2", "2_3", "2_4", "3_2", "3_3", "3_4"],
            "val": ["1_5", "2_5", "3_5"],
            "test": ["1_1", "2_1", "3_1"],
        },
    },
    "domain_shift": {
        "fold_1": {
            "train": ["1_2", "1_3", "1_4", "1_5", "2_2", "2_3", "2_4", "2_5"],
            "val": ["1_1", "2_1"],
            "test": ["3_1", "3_2", "3_3", "3_4", "3_5"],
        },
        "fold_2": {
            "train": ["1_1", "1_3", "1_4", "1_5", "2_1", "2_3", "2_4", "2_5"],
            "val": ["1_2", "2_2"],
            "test": ["3_1", "3_2", "3_3", "3_4", "3_5"],
        },
        "fold_3": {
            "train": ["1_1", "1_2", "1_4", "1_5", "2_1", "2_2", "2_4", "2_5"],
            "val": ["1_3", "2_3"],
            "test": ["3_1", "3_2", "3_3", "3_4", "3_5"],
        },
        "fold_4": {
            "train": ["1_1", "1_2", "1_3", "1_5", "2_1", "2_2", "2_3", "2_5"],
            "val": ["1_4", "2_4"],
            "test": ["3_1", "3_2", "3_3", "3_4", "3_5"],
        },
        "fold_5": {
            "train": ["1_1", "1_2", "1_3", "1_4", "2_1", "2_2", "2_3", "2_4"],
            "val": ["1_5", "2_5"],
            "test": ["3_1", "3_2", "3_3", "3_4", "3_5"],
        },
    },
}


class XJTU_SYLoader(PHMDMultiSourceLoader):
    """
    XJTU-SY PHMD datasource using the refactored PHMD base.

    Parameters
    ----------
    split_mode : str, optional
        Evaluation protocol to activate (e.g. ``"in_domain"``, ``"domain_shift"``,
        ``"phmd_split"``). When ``None`` the raw PHMD fold split is used unchanged.
    fold_id : int or str, optional
        Fold identifier within a CV split table. Required when *split_mode* maps to
        a multi-fold structure.
    split_assignments : dict, optional
        Explicit ``{split: [unit_keys]}`` table. Takes precedence over *split_mode*.
    split_assignments_by_mode : dict, optional
        Full mode-keyed split table. Falls back to the module constant when ``None``.
    **kwargs
        Forwarded to :class:`PHMDMultiSourceLoader`.
    """

    def __init__(
        self,
        split_mode: str | None = None,
        fold_id: int | str | None = None,
        split_assignments: dict[str, list[str]] | None = None,
        split_assignments_by_mode: dict[str, dict[str, Any]] | None = None,
        **kwargs,
    ):
        """
        Initialize the loader and resolve the active split table.

        Parameters
        ----------
        split_mode : str, optional
            Evaluation protocol to activate (e.g. ``"in_domain"``, ``"domain_shift"``,
            ``"phmd_split"``). When ``None`` the raw PHMD fold split is used unchanged.
        fold_id : int or str, optional
            Fold identifier within a CV split table. Required when *split_mode* maps to
            a multi-fold structure.
        split_assignments : dict, optional
            Explicit ``{split: [unit_keys]}`` table. Takes precedence over *split_mode*.
        split_assignments_by_mode : dict, optional
            Full mode-keyed split table. Falls back to the module constant when ``None``.
        **kwargs
            Forwarded to :class:`PHMDMultiSourceLoader`.
        """
        # Preserve the custom split configuration in the constructor snapshot so
        # payload-cache fingerprints change when the evaluation protocol changes.
        kwargs.update(
            split_mode=split_mode,
            fold_id=fold_id,
            split_assignments=deepcopy(split_assignments),
            split_assignments_by_mode=deepcopy(split_assignments_by_mode),
        )
        super().__init__(**kwargs)
        self.split_mode = split_mode
        self.fold_id = fold_id
        self.split_assignments = self._resolve_split_assignments(
            split_assignments=split_assignments,
            split_mode=split_mode,
            fold_id=fold_id,
            split_assignments_by_mode=split_assignments_by_mode,
        )

    def _resolve_split_assignments(
        self,
        *,
        split_assignments: dict[str, list[str]] | None,
        split_mode: str | None,
        fold_id: int | str | None,
        split_assignments_by_mode: dict[str, dict[str, Any]] | None,
    ) -> dict[str, list[str]] | None:
        """
        Return the active unit split table for the configured evaluation mode.

        When no custom split mode is provided, the loader falls back to the PHMD
        fold split exactly as before.

        Parameters
        ----------
        split_assignments : dict, optional
            Explicit split table; takes precedence over all other arguments.
        split_mode : str, optional
            Name of the evaluation mode to look up.
        fold_id : int or str, optional
            Fold key within a CV split table.
        split_assignments_by_mode : dict, optional
            Mode-keyed split table; falls back to the module constant when ``None``.

        Returns
        -------
        dict or None
            Resolved ``{split: [unit_keys]}`` mapping, or ``None`` if no split mode
            is configured.
        """
        if split_assignments is not None:
            return {
                split: list(unit_keys) for split, unit_keys in split_assignments.items()
            }

        if split_mode is None:
            return None

        split_tables = split_assignments_by_mode or XJTU_SY_SPLIT_ASSIGNMENTS_BY_MODE
        if split_mode not in split_tables:
            raise ValueError(
                "Unknown XJTU-SY split_mode "
                f"{split_mode!r}. Available modes: {sorted(split_tables)}"
            )

        split_table_or_folds = split_tables[split_mode]
        if {"train", "val", "test"}.issubset(split_table_or_folds):
            resolved_table = split_table_or_folds
        else:
            if fold_id is None:
                raise ValueError(
                    "XJTU_SYLoader requires fold_id when split_assignments_by_mode "
                    "contains multiple CV folds."
                )
            fold_key = (
                fold_id
                if isinstance(fold_id, str) and str(fold_id) in split_table_or_folds
                else f"fold_{fold_id}"
            )
            if fold_key not in split_table_or_folds:
                raise ValueError(
                    "Unknown XJTU-SY fold_id "
                    f"{fold_id!r} for split_mode {split_mode!r}. "
                    f"Available folds: {sorted(split_table_or_folds)}"
                )
            resolved_table = split_table_or_folds[fold_key]
        return {split: list(unit_keys) for split, unit_keys in resolved_table.items()}

    def _load_data(self):
        """
        Load PHMD payload and apply custom split assignments if configured.

        Returns
        -------
        dict
            Payload dict keyed by data field and split name.
        """
        out_dict = super()._load_data()
        if self.split_assignments is None:
            return out_dict

        remapped = self._apply_custom_split_assignments(out_dict)
        self._refresh_split_metadata(remapped)
        return remapped

    def _apply_custom_split_assignments(self, out_dict):
        """
        Reassemble the PHMD unit payload using the configured custom protocol.

        PHMD already materializes one payload entry per bearing, so custom split
        support only needs to relocate whole-unit records across train/val/test.

        Parameters
        ----------
        out_dict : dict
            Raw payload dict from the PHMD loader, keyed by data field and split.

        Returns
        -------
        dict
            Remapped payload with units relocated across train/val/test splits.
        """
        assert self.split_assignments is not None, "Custom split assignments missing."
        unit_metadata = out_dict["unit_metadata"]

        indexed_units: dict[str, tuple[str, int]] = {}
        for split_name in ("train", "val", "test"):
            for idx, meta in enumerate(unit_metadata.get(split_name, [])):
                unit_key = meta["unit_name"].replace("Bearing ", "")
                indexed_units[unit_key] = (split_name, idx)

        self._validate_split_assignments(indexed_units)

        remapped = {key: {"train": [], "val": [], "test": []} for key in out_dict}
        for target_split in ("train", "val", "test"):
            for unit_key in self.split_assignments.get(target_split, []):
                source_split, idx = indexed_units[unit_key]
                for key in out_dict:
                    remapped[key][target_split].append(out_dict[key][source_split][idx])
        return remapped

    def _validate_split_assignments(
        self, indexed_units: dict[str, tuple[str, int]]
    ) -> None:
        """
        Validate that configured split assignments are consistent with the PHMD payload.

        Parameters
        ----------
        indexed_units : dict
            Mapping of unit key to ``(split_name, index)`` from the PHMD payload.
        """
        configured_keys: list[str] = []
        for split_name in ("train", "val", "test"):
            configured_keys.extend(self.split_assignments.get(split_name, []))

        duplicates = sorted(
            {key for key in configured_keys if configured_keys.count(key) > 1}
        )
        if duplicates:
            raise ValueError(f"Units assigned to multiple splits: {duplicates}")

        unknown_units = sorted(set(configured_keys) - set(indexed_units))
        if unknown_units:
            raise ValueError(
                f"Configured units not found in PHMD fold payload: {unknown_units}"
            )

        unassigned_units = sorted(set(indexed_units) - set(configured_keys))
        if unassigned_units:
            raise ValueError(
                f"Unassigned units present in PHMD fold payload: {unassigned_units}"
            )

    def _refresh_split_metadata(self, out_dict) -> None:
        """
        Refresh unit name and ID metadata after custom split reassignment.

        Parameters
        ----------
        out_dict : dict
            Remapped payload dict after custom split assignments have been applied.
        """
        unit_metadata = out_dict.get("unit_metadata", {})
        unit_ids = out_dict.get("unit_id", {})
        self.meta_data["unit_names"] = {
            split: [meta["unit_name"] for meta in unit_metadata.get(split, [])]
            for split in ("train", "val", "test")
        }
        self.meta_data["unit_ids"] = {
            split: unit_ids.get(split, []) for split in ("train", "val", "test")
        }

    def _get_features_columns(self):
        """
        Return the list of feature column names from loader metadata.

        Returns
        -------
        list
            Feature column names.
        """
        return self.meta_data["features"]

    def _get_target_column(self):
        """
        Return the target column name for the current task mode.

        Returns
        -------
        str
            Target column name.
        """
        return self.task_mode

    def _get_unit_column(self):
        """
        Return the column name used to identify individual bearing units.

        Returns
        -------
        str
            Unit column name.
        """
        return "bearing"  # TODO: self.meta_data["identifier"] do not work correctly for fault

    def _preprocess_fold(self, main_fold, auxiliary_tasks_fold_dict):
        """
        Merge auxiliary task data into a PHMD fold when fault task mode is active.

        Parameters
        ----------
        main_fold : dict
            Primary task fold data.
        auxiliary_tasks_fold_dict : dict
            Auxiliary task fold data keyed by task name.

        Returns
        -------
        dict
            Merged fold data.
        """
        if self.task_mode == "fault":
            assert (
                "rul" in list(auxiliary_tasks_fold_dict.keys())
            ), "When using fault as main task, please provide rul as auxiliary task in the datasource"
            rul_fold = auxiliary_tasks_fold_dict["rul"]
            fault_fold = main_fold
            # Merge the rul information into the fault dataframe
            return merge_data_folds_on_key(rul_fold, fault_fold)
        else:
            return main_fold

    def _process_unit(
        self, df_unit: pd.DataFrame, unit_name: str, features_col: list, target_col: str
    ):
        """
        Extract features, targets, and metadata from a single unit's DataFrame.

        This internal method handles the logic for two primary task modes:
        1.  'rul': Remaining Useful Life prediction.
        2.  'fault': Fault diagnosis (multi-label classification).

        It also supports a 'ragged' mode, where continuous time-series data is
        reshaped into fixed-size sequences, creating a ragged array structure.

        Parameters
        ----------
        df_unit : pd.DataFrame
            The DataFrame containing data for a single unit.
        unit_name : str
            The identifier for the unit (e.g., 'Bearing1_3_1').
        features_col : list
            A list of column names to be used as features.
        target_col : str
            The name of the primary target column (used for RUL).

        Returns
        -------
        dict
            A dictionary containing the processed 'features', 'target', and
            'metadata' for the unit.

        Raises
        ------
        ValueError
            If an unknown 'task_mode' is specified or if data length
            is incompatible with the ragged processing mode.
        NotImplementedError
            If ragged mode is attempted with the 'fault' task.
        """
        # --- 1. Prepare Features and Targets based on Task Mode ---

        # Use .copy() to prevent SettingWithCopyWarning from pandas
        df_features = df_unit[features_col].copy().reset_index(drop=True)

        # Process targets differently depending on the task
        if self.task_mode == "rul":
            df_target = df_unit[[target_col]].copy().reset_index(drop=True)
            processed_target_col = target_col

        elif self.task_mode == "fault":
            # For fault diagnosis, the target is a multi-label binary array
            df_target = df_unit[FAULT_TARGET_COLS].copy().to_numpy()
            processed_target_col = FAULT_TARGET_COLS

        else:
            raise ValueError(f"Unknown task_mode: {self.task_mode}")

        # --- 2. Handle Ragged Array Processing (if enabled) ---

        if self.use_ragged:
            # For ragged mode, we reshape the flat time-series data into sequences
            # of a fixed length (SAMPLE_RANGE_SIZE).

            # Ensure the total number of samples is divisible by the sequence length
            if len(df_features) % SAMPLE_RANGE_SIZE != 0:
                raise ValueError(
                    f"Unit {unit_name} has {len(df_features)} samples, which is not "
                    f"divisible by the required sample range of {SAMPLE_RANGE_SIZE}."
                )

            # Reshape features from (total_samples, num_features) to
            # (num_sequences, sequence_length, num_features)
            df_features = df_features.to_numpy().reshape(
                -1, SAMPLE_RANGE_SIZE, df_features.shape[-1]
            )

            if self.task_mode == "rul":
                # Reshape target to match the new sequence structure
                df_target = df_target.to_numpy().reshape(-1, SAMPLE_RANGE_SIZE, 1)

                # Integrity Check: The RUL value should be constant across one entire sequence.
                # This check broadcasts the first RUL value of each sequence and compares
                # it against all other values in that same sequence.
                if not np.all(df_target == df_target[:, 0:1, :]):
                    raise ValueError(
                        f"Unit {unit_name} has varying RUL values within a single sample range of "
                        f"{SAMPLE_RANGE_SIZE}. Please check data integrity."
                    )

                # Since RUL is constant per sequence, we only need one value.
                # This reduces the target from shape (n, 32768, 1) to (n, 1, 1).
                # df_target = df_target[:, 0:1, :]

                # Finally, flatten to a simple 2D array of shape (num_sequences, 1)
                # df_target = df_target.reshape(-1, 1)

            elif self.task_mode == "fault":
                # This functionality has not been implemented yet.
                raise NotImplementedError(
                    "Ragged mode is not yet implemented for the 'fault' task yet."
                )

            # Convert the NumPy arrays to Awkward Arrays for ragged processing
            df_features = ak.from_regular(ak.from_numpy(df_features), axis=1)
            df_target = ak.from_numpy(df_target)

        # --- 3. Assemble and Return the Final Dictionary ---
        unit_id = UNIT_NAMES_TO_ID.get(unit_name.strip("Bearing"), -1)
        return {
            "features": df_features,
            "target": df_target,
            "unit_id": ak.from_numpy(np.array(unit_id)),
            "metadata": {
                "unit_id": unit_id,
                "unit_name": f"Bearing {unit_name.strip('Bearing')}",
                "unit_length": len(df_features),
                "features_columns": features_col,
                "target_col": processed_target_col,
                "target_in_features": True,
                "task": self.task_mode,
            },
        }


def merge_data_folds_on_key(main_fold, aux_fold, key="bearing", how="left"):
    """
    Merge dataframes from an auxiliary dictionary into a main dictionary on a key.

    This function iterates through the splits ('train', 'val', etc.) in the
    main_fold, finds the corresponding split in the aux_fold, and performs a
    merge based on the specified key.

    Parameters
    ----------
    main_fold : dict
        Dictionary of data splits (e.g., {'train': pd.DataFrame})
        to which data will be added.
    aux_fold : dict
        Dictionary of data splits containing the auxiliary
        information to merge.
    key : str
        The column name to merge on. Defaults to 'bearing'.
    how : str
        The type of merge ('left', 'inner', etc.). Defaults to 'left'.

    Returns
    -------
    dict
        A new dictionary containing the merged DataFrames.
    """
    merged_folds = {}  # Create a new dictionary for the results

    # Iterate through each split ('train', 'val', etc.) in the main dictionary
    for split_name, main_df in main_fold.items():
        # Check if the corresponding split exists in the auxiliary dictionary
        if split_name in aux_fold:
            aux_df = aux_fold[split_name]

            # 1. Identify the new columns to be added from the auxiliary dataframe.
            # This avoids duplicating columns that already exist in main_df.
            new_columns = [col for col in aux_df.columns if col not in main_df.columns]

            # If there are no new columns to add, skip the merge for this split
            if not new_columns:
                merged_folds[split_name] = main_df.copy()
                continue

            # 2. Create a clean lookup table from the auxiliary data.
            # This ensures we have one unique row of info per key.
            lookup_columns = [key] + new_columns
            aux_lookup_df = aux_df[lookup_columns].drop_duplicates(subset=[key])

            # 3. Merge the main dataframe with the new lookup table
            merged_folds[split_name] = pd.merge(main_df, aux_lookup_df, on=key, how=how)
        else:
            # If the split doesn't exist in the aux_fold, just copy the original df
            merged_folds[split_name] = main_df.copy()

    return merged_folds
