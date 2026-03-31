"""
Provide datasource helpers for phmd pronostia.

"""

from __future__ import annotations

import logging

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
    "1_6": (1, 6),
    "1_7": (1, 7),
    "2_1": (2, 1),
    "2_2": (2, 2),
    "2_3": (2, 3),
    "2_4": (2, 4),
    "2_5": (2, 5),
    "2_6": (2, 6),
    "2_7": (2, 7),
    "3_1": (3, 1),
    "3_2": (3, 2),
    "3_3": (3, 3),
}


TEST_RULS = {
    "1_3": 5730,
    "1_4": 2900,  # 339, Adjusting this then Table 4 from the "Remaining useful life estimation of bearing via temporal convolutional networks enhanced by a gated convolutional unit" paper make sense
    "1_5": 1610,
    "1_6": 1460,
    "1_7": 7570,
    "2_3": 7530,
    "2_4": 1390,
    "2_5": 3090,
    "2_6": 1290,
    "2_7": 580,
    "3_3": 820,
}
SAMPLE_RANGE_SIZE = 2560


class PronostiaLoader(PHMDMultiSourceLoader):
    """Pronostia PHMD datasource using the refactored PHMD base."""

    def _get_features_columns(self):
        return self.meta_data["features"]

    def _get_target_column(self):
        return self.task_mode

    def _get_unit_column(self):
        return self.meta_data["identifier"]

    def _preprocess_fold(self, main_fold, auxiliary_tasks_fold_dict):
        return main_fold

    def _process_unit(
        self, df_unit: pd.DataFrame, unit_name: str, features_col: list, target_col: str
    ):
        """
        Processes a single unit's DataFrame for Remaining Useful Life (RUL) prediction.

        This method extracts features and targets. It supports two modes:
        1.  Standard Mode: Returns features and targets as pandas DataFrames.
        2.  Ragged Mode: Reshapes continuous data into fixed-size sequences,
            creating ragged Awkward Arrays suitable for sequence models.

        Parameters
        ----------
        df_unit : pd.DataFrame
            The DataFrame for a single operational unit.
        unit_name : str
            The identifier for the unit (e.g., 'Bearing1_3').
        features_col : list
            A list of column names to be used as features.
        target_col : str
            The name of the RUL target column.

        Returns
        -------
        dict
            A dictionary containing the processed 'features', 'target', and 'metadata'.

        Raises
        ------
        ValueError
            If data length is incompatible with the ragged processing mode,
            or if RUL values are inconsistent within a sequence in ragged mode.
        """
        # --- 1. Prepare Features and Target ---

        # Ensure the task mode is set to 'rul' as this function is specific to it.
        if self.task_mode != "rul":
            raise ValueError(
                f"This processor is configured for 'rul' task, but got '{self.task_mode}'."
            )

        # Use .copy() to prevent pandas' SettingWithCopyWarning
        features = df_unit[features_col].copy().reset_index(drop=True)
        target = df_unit[[target_col]].copy().reset_index(drop=True)

        # --- 2. Handle Ragged Array Processing (if enabled) ---

        if self.use_ragged:
            # Check if the total number of samples can be evenly divided into sequences.
            if len(features) % SAMPLE_RANGE_SIZE != 0:
                raise ValueError(
                    f"Unit {unit_name} has {len(features)} samples, which is not "
                    f"divisible by the required sample range of {SAMPLE_RANGE_SIZE}."
                )

            # Reshape features from (total_samples, num_features) into
            # (num_sequences, sequence_length, num_features).
            features = features.to_numpy().reshape(
                -1, SAMPLE_RANGE_SIZE, features.shape[-1]
            )

            # Reshape target to match the new sequence structure.
            target = target.to_numpy().reshape(-1, SAMPLE_RANGE_SIZE, 1)

            # Data Integrity Check: The RUL value must be constant across any single sequence.
            # This check works by comparing every RUL value in a sequence to the first one.
            if not np.unique(target, axis=1).shape[1] == 1:
                raise ValueError(
                    f"Unit {unit_name} has wrong RUL values within a single sample range"
                )

            # Since RUL is constant per sequence, we only need one value per sequence.
            # This reduces the target from (num_sequences, 32768, 1) to (num_sequences, 1).
            # target = target[:, 0, :].reshape(-1, 1)

            # Convert the NumPy arrays to Awkward Arrays for ragged processing.
            features = ak.from_regular(ak.from_numpy(features), axis=1)
            target = ak.from_numpy(target)

        # --- 3. Assemble and Return the Final Dictionary ---

        # Use .get() for safer dictionary access to avoid errors on missing keys.

        unit_key = unit_name.strip("Bearing")
        unit_id = UNIT_NAMES_TO_ID.get(unit_key, -1)

        # It seems to be not needed
        # if self.cut_the_test and unit_key in TEST_RULS:
        #     first_predicting_point = int(TEST_RULS[unit_key] / 10)
        #     features = features[:-first_predicting_point]
        #     target = target[:-first_predicting_point]

        return {
            "features": features,
            "target": target,
            "unit_id": ak.from_numpy(np.array(unit_id)),
            "metadata": {
                "unit_id": unit_id,
                "unit_name": f"Bearing {unit_key}",
                "unit_length": len(
                    features
                ),  # This will be num_sequences in ragged mode
                "features_columns": features_col,
                "target_col": target_col,
                "target_in_features": True,
                "task": self.task_mode,
            },
        }
