"This code is taken from https://github.com/MichaelBosello/battery-rul-estimation/tree/main"

import numpy as np
import pandas as pd
import logging
import plotly.graph_objects as go
import scipy.io
from pathlib import Path
from phmd.download import download

logger = logging.getLogger(__name__)

DATA_PATH = "data/nasa-randomized/"
NOMINAL_CAPACITY = 2.2


class NasaRandomizedData:
    def __init__(self, data_path, force_download=False):
        download("NB14", cache_dir=data_path, force=force_download, unzip=True)
        self.path = Path(data_path).expanduser() / "datasets/NB14"

    def get_discharge_whole_cycle_future(
        self, train_names, test_names, validation_names=None
    ):
        logger.info("Loading train data...")
        (
            train_x,
            train_y,
            battery_n_cycle_train,
            time_train,
            current_train,
            initial_cycle_lenghts_train,
        ) = self._get_data(train_names)
        logger.info("Loading test data...")
        (
            test_x,
            test_y,
            battery_n_cycle_test,
            time_test,
            current_test,
            initial_cycle_lenghts_test,
        ) = self._get_data(test_names)
        if validation_names is not None:
            logger.info("Loading validation data...")
            (
                validation_x,
                validation_y,
                battery_n_cycle_validation,
                time_validation,
                current_validation,
                initial_cycle_lenghts_validation,
            ) = self._get_data(validation_names)

        logger.info(
            """Train x: %s, train y soh: %s | Test x: %s, test y soh: %s | 
                            battery n cycle train: %s, battery n cycle test: %s, 
                            time train: %s, time test: %s |
                            raw current train: %s, raw current test: %s |
                            """
            % (
                train_x.shape,
                train_y.shape,
                test_x.shape,
                test_y.shape,
                battery_n_cycle_train.shape,
                battery_n_cycle_test.shape,
                time_train.shape,
                time_test.shape,
                current_train.shape,
                current_test.shape,
            )
        )
        if validation_names is not None:
            logger.info(
                """Validation x: %s, validation y soh: %s | 
                                battery n cycle validation: %s, 
                                time validation: %s ,
                                raw current validation: %s |
                                """
                % (
                    validation_x.shape,
                    validation_y.shape,
                    battery_n_cycle_validation.shape,
                    time_validation.shape,
                    current_validation.shape,
                )
            )
            return (
                train_x,
                train_y,
                validation_x,
                validation_y,
                test_x,
                test_y,
                battery_n_cycle_train,
                battery_n_cycle_validation,
                battery_n_cycle_test,
                time_train,
                time_validation,
                time_test,
                current_train,
                current_validation,
                current_test,
                initial_cycle_lenghts_train,
                initial_cycle_lenghts_validation,
                initial_cycle_lenghts_test,
            )

        return (
            train_x,
            train_y,
            test_x,
            test_y,
            battery_n_cycle_train,
            battery_n_cycle_test,
            time_train,
            time_test,
            current_train,
            current_test,
            initial_cycle_lenghts_train,
            initial_cycle_lenghts_test,
        )

    def _get_data(self, names):
        cycle_x = []
        cycle_y = []
        first_y = True
        y_between_count = 0
        battery_n_cycle = []
        time = []
        current = []
        n_cycles = 0
        max_step = 0
        for name in names:
            logger.info("Processing file %s" % name)
            path_str = str(self.path / name)
            raw_data = scipy.io.loadmat(path_str)["data"][0][0][0][0]
            cycle = pd.DataFrame(raw_data)

            cycle_num = 0
            cycle["cycle"] = cycle_num
            current_type = cycle.loc[0, "type"]
            for index in range(1, len(cycle.index)):
                if (
                    (current_type == "C" and cycle.loc[index, "type"] == "D")
                    or (current_type == "D" and cycle.loc[index, "type"] == "C")
                    or (current_type == "R" and cycle.loc[index, "type"] != "R")
                ):
                    current_type = cycle.loc[index, "type"]
                    cycle_num += 1
                cycle.loc[index, "cycle"] = cycle_num

            for x in set(cycle["cycle"]):
                if cycle.loc[cycle["cycle"] == x, "type"].iloc[0] != "D":
                    continue

                cycle_x.append(
                    np.column_stack(
                        [
                            np.hstack(
                                cycle.loc[cycle["cycle"] == x, "voltage"]
                                .to_numpy()
                                .flatten()
                            ).flatten(),
                            np.hstack(
                                cycle.loc[cycle["cycle"] == x, "current"]
                                .to_numpy()
                                .flatten()
                            ).flatten(),
                            np.hstack(
                                cycle.loc[cycle["cycle"] == x, "temperature"]
                                .to_numpy()
                                .flatten()
                            ).flatten(),
                        ]
                    )
                )

                n_cycles += 1
                step_time = np.hstack(
                    cycle.loc[cycle["cycle"] == x, "time"].to_numpy().flatten()
                ).flatten()
                time.append(step_time / 3600)
                current.append(
                    np.hstack(
                        cycle.loc[cycle["cycle"] == x, "current"].to_numpy().flatten()
                    ).flatten()
                )
                max_step = max([max_step, cycle_x[-1].shape[0]])

                if cycle.loc[cycle["cycle"] == x, "comment"].iloc[
                    0
                ] == "reference discharge" and (
                    x < 2
                    or cycle.loc[cycle["cycle"] == x - 2, "comment"].iloc[0]
                    != "reference discharge"
                ):
                    current_y = (
                        np.trapz(
                            current[-1],
                            np.hstack(
                                cycle.loc[cycle["cycle"] == x, "time"]
                                .to_numpy()
                                .flatten()
                            ).flatten(),
                        )
                        / 3600
                    )
                    if y_between_count > 0:
                        step_y = (cycle_y[-1] - current_y) / y_between_count
                        while y_between_count > 0:
                            cycle_y.append(cycle_y[-1] - step_y)
                            y_between_count -= 1
                    cycle_y.append(current_y)
                elif first_y is True:
                    cycle_y.append(NOMINAL_CAPACITY)
                else:
                    y_between_count += 1
                first_y = False

            while y_between_count > 0:
                cycle_y.append(cycle_y[-1])
                y_between_count -= 1
            first_y = True
            battery_n_cycle.append(n_cycles)

        cycle_x, initial_cycle_lenghts = self._to_padded_numpy(
            cycle_x, [len(cycle_x), max_step, len(cycle_x[0][0])]
        )
        cycle_y = np.array(cycle_y)
        battery_n_cycle = np.array(battery_n_cycle)
        time, _ = self._to_padded_numpy(time, [len(time), max_step])
        current, _ = self._to_padded_numpy(current, [len(current), max_step])

        return cycle_x, cycle_y, battery_n_cycle, time, current, initial_cycle_lenghts

    def _to_padded_numpy(self, l, shape):
        padded_array = np.zeros(shape)
        initia_lenghts = []
        for i, j in enumerate(l):
            padded_array[i][0 : len(j)] = j
            initia_lenghts.append(len(j))
        return padded_array, initia_lenghts
