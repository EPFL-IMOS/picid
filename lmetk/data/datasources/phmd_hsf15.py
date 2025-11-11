import logging

import numpy as np
import awkward as ak

from lmetk.data.datasources.base.phmd_loader import PHMDMultiSourceLoader

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
        features = ak.unflatten(df_features.values, counts)
        targets = ak.unflatten(df_target.values, counts)

        # Not sure how we want to train here, by cycle or within cycle.
        # I think within cycle is better due to limited data.
        # for t in targets:
        #     assert (
        #         len(np.unique(t.to_numpy())) == 1
        #     ), "Target values within a cycle should be identical."

        # targets = targets[:, 0]

        return {
            "features": features,
            "time_features": np.array(df_features.index),
            "unit_id": "no_id_available",
            "health_states": targets,
            "metadata": {
                "unit_id": "no_id_available",
                "unit_name": unit_name,
                "unit_length": len(df_features),
                "features_columns": features_col,
                "target_col": target_col,
                "target_in_the_featurs": True,
            },
        }
