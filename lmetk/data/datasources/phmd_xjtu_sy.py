import logging
import pandas as pd
import numpy as np
import awkward as ak
from lmetk.data.datasources.base.phmd_loader import PHMDMultiSourceLoader
from lmetk.data.datasources.nb14.utils import table_to_ak_array

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


class XJTU_SYLoader(PHMDMultiSourceLoader):
    """XJTU SY Data Loader"""

    def _get_features_columns(self):
        return self.meta_data["features"]

    def _get_target_column(self):
        return self.task_mode

    def _get_unit_column(self):
        return "bearing"  # TODO: self.meta_data["identifier"] do not work correctly for fault

    def _preprocess_fold(self, main_fold, auxiliary_tasks_fold_dict):
        if self._init_kwargs["task_mode"] == "fault":
            assert "rul" in list(
                auxiliary_tasks_fold_dict.keys()
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
        Processes a single unit's DataFrame to extract features, targets, and metadata.

        This internal method handles the logic for two primary task modes:
        1.  'rul': Remaining Useful Life prediction.
        2.  'fault': Fault diagnosis (multi-label classification).

        It also supports a 'ragged' mode, where continuous time-series data is
        reshaped into fixed-size sequences, creating a ragged array structure.

        Args:
            df_unit (pd.DataFrame): The DataFrame containing data for a single unit.
            unit_name (str): The identifier for the unit (e.g., 'Bearing1_3_1').
            features_col (list): A list of column names to be used as features.
            target_col (str): The name of the primary target column (used for RUL).

        Returns:
            dict: A dictionary containing the processed 'features', 'target', and
                'metadata' for the unit.

        Raises:
            ValueError: If an unknown 'task_mode' is specified or if data length
                        is incompatible with the ragged processing mode.
            NotImplementedError: If ragged mode is attempted with the 'fault' task.
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

        if self._init_kwargs["use_ragged"]:
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
                df_target = df_target[:, 0:1, :]

                # Finally, flatten to a simple 2D array of shape (num_sequences, 1)
                df_target = df_target.reshape(-1, 1)

            elif self.task_mode == "fault":
                # This functionality has not been implemented yet.
                raise NotImplementedError(
                    "Ragged mode is not yet implemented for the 'fault' task."
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
                "unit_name": f"Bearing {unit_name.strip("Bearing")}",
                "unit_length": len(df_features),
                "features_columns": features_col,
                "target_col": processed_target_col,
                "target_in_features": True,  # Fixed typo: 'featurs' -> 'features'
                "task": self.task_mode,
            },
        }


def merge_data_folds_on_key(main_fold, aux_fold, key="bearing", how="left"):
    """
    Merges dataframes from an auxiliary dictionary into a main dictionary on a key.

    This function iterates through the splits ('train', 'val', etc.) in the
    main_fold, finds the corresponding split in the aux_fold, and performs a
    merge based on the specified key.

    Args:
        main_fold (dict): Dictionary of data splits (e.g., {'train': pd.DataFrame})
                          to which data will be added.
        aux_fold (dict): Dictionary of data splits containing the auxiliary
                         information to merge.
        key (str): The column name to merge on. Defaults to 'bearing'.
        how (str): The type of merge ('left', 'inner', etc.). Defaults to 'left'.

    Returns:
        dict: A new dictionary containing the merged DataFrames.
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
