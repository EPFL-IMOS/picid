import logging

import numpy as np
import awkward as ak

from picid.data.datasources.base.phmd_loader import PHMDMultiSourceLoader

logger = logging.getLogger(__name__)


class HSF15Loader(PHMDMultiSourceLoader):
    def _get_features_columns(self):
        return self.meta_data["features"]

    def _get_target_column(self):
        return self.task_mode

    def _get_unit_column(self):
        return "unit_name"  # we have just one unit

    def _preprocess_fold(self, main_fold, auxiliary_tasks_fold_dict):
        # we have just one unit
        for key in main_fold.keys():
            main_fold[key]["unit_name"] = "HSF15_Unit_1"
        return main_fold

    def _process_unit(self, df_unit, unit_name, features_col, target_col):
        cycle_col = df_unit["cycle"].copy().reset_index(drop=True)
        df_features = df_unit[features_col].copy().reset_index(drop=True)
        df_target = df_unit[target_col].copy().reset_index(drop=True)

        _, idx, counts = np.unique(
            cycle_col.values, return_index=True, return_counts=True
        )

        is_monotonic = np.all(cycle_col.values[:-1] <= cycle_col.values[1:])
        if not is_monotonic:
            raise ValueError(
                "Cycle column is not monotonically increasing; grouping by offsets would be invalid."
            )

        # Check if reordering occurred
        order = np.argsort(idx)
        if not np.array_equal(order, np.arange(len(order))):
            raise ValueError("Cycles are not in order. This should not happen.")

        # We can make it work anyways, but better fail above and make the user check.
        counts = counts[order]
        features = ak.unflatten(df_features.values, counts)  # eventually (N, 6000, F)
        # targets = ak.unflatten(df_target.values, counts)

        targets = ak.Array(
            [i for i in df_target.values.reshape(-1, 6000, 1)]
        )  # df_target.values.reshape(-1, 6000, 1) --> (N,6000,1)
        targets = ak.to_regular(targets, axis=-1)

        # Eventually features have N cycles, C cycle lengths, F features --> (N,C,F)  ["ce", "cp","eps1", "fs1", "fs2", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "se", "ts1", "ts2", "ts3", "ts4", "vs1"]
        # Targets have N cycles, C cycle lengths, 1 --> (N,C,1), where the last dim is a class, and the class is repeated C times
        # CP,CE have 100Hz sampling.
        # Cycle duration is 60 sec. --> 6000 (measurenents) * 100Hz = 60 samples per cycle, 1 per each second
        # Hence every entry of features CP,CE populated to match the cycle length of 60 sec * 100Hz = 6000 measurements
        # (EPS1, )
        return {
            "features": features,
            "time_features": np.array(df_features.index),
            "unit_id": "no_id_available",
            "target": targets,
            "metadata": {
                "unit_id": "no_id_available",
                "unit_name": unit_name,
                "unit_length": len(df_features),
                "features_columns": features_col,
                "target_col": target_col,
                "target_in_the_featurs": True,
            },
        }
