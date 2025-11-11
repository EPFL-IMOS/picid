import logging

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .unibo_powertools_data import CapacityCols, CycleCols


import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ModelDataHandler:
    def __init__(self, dataset, x_indices, scaler_type=MinMaxScaler):
        self.logger = logging.getLogger()
        self.dataset = dataset
        self.x_indices = x_indices
        self.scaler_type = scaler_type

        # === REFACTORED __init__ ===
        # 1. Get the new dictionaries from the dataset
        (self.charge_cyc_all, self.charge_cap_all) = self.dataset.get_charge_data()
        (self.discharge_cyc_all, self.discharge_cap_all) = (
            self.dataset.get_discharge_data()
        )

        # 2. Assign attributes for backward compatibility with other methods
        self.train_charge_cyc = self.charge_cyc_all.get("train")
        self.train_charge_cap = self.charge_cap_all.get("train")
        self.test_charge_cyc = self.charge_cyc_all.get("test")
        self.test_charge_cap = self.charge_cap_all.get("test")

        self.train_discharge_cyc = self.discharge_cyc_all.get("train")
        self.train_discharge_cap = self.discharge_cap_all.get("train")
        self.test_discharge_cyc = self.discharge_cyc_all.get("test")
        self.test_discharge_cap = self.discharge_cap_all.get("test")

        # Also assign validation data for the new _future method
        self.val_discharge_cyc = self.discharge_cyc_all.get("val")
        self.val_discharge_cap = self.discharge_cap_all.get("val")

        # This will now work correctly, as it relies on self.train_charge_cyc
        # self.__assign_scalers()

    def __assign_scalers(self):
        self.charge_scalers = self.__create_scalers(self.train_charge_cyc)
        self.discharge_scalers = self.__create_scalers(self.train_discharge_cyc)

    def __create_scalers(self, cyc):
        scalers = []
        for index in self.x_indices:
            scalers.append(self.__create_scaler(cyc, index))
        return scalers

    def __create_scaler(self, cyc, col_index):
        data = np.concatenate(cyc)[:, col_index].reshape(-1, 1)
        scaler_x = self.scaler_type()
        scaler_x.fit_transform(data)
        return scaler_x

    def get_scalers(self):
        return self.charge_scalers, self.discharge_scalers

    def get_discharge_whole_cycle(
        self, output_capacity=False, multiple_output=False, soh=False
    ):
        """x: [ [[voltage, current, temperature], ...], ...] \n
        SOH y (single step): [ [soh/last_charging_capacity, remaining_time_to_cell_end], ... ] \n
        SOH y (multiple steps): [ [[soh/last_charging_capacity, remaining_time_to_cell_end], ...], ... ]\n
        SOC y: [[[soc/discharging_capacity, remaining_time_to_cycle_end], ...], ...]"""

        if soh:
            y_indices = [
                (
                    CapacityCols.CORRESPONDING_CHARGING_CAPACITY
                    if output_capacity
                    else CapacityCols.SOH
                ),
                CapacityCols.REMAINING_TIME_TO_CELL_END,
            ]
            train_raw_x, train_y = self.__get_whole_cycle_soh_x_y(
                self.train_discharge_cyc,
                self.train_discharge_cap,
                self.x_indices,
                y_indices,
            )
            test_raw_x, test_y = self.__get_whole_cycle_soh_x_y(
                self.test_discharge_cyc,
                self.test_discharge_cap,
                self.x_indices,
                y_indices,
            )
        else:
            y_indices = [
                CycleCols.DISCHARGING_CAPACITY if output_capacity else CycleCols.SOC,
                CycleCols.REMAINING_TIME_TO_CYCLE_END,
            ]
            train_raw_x, train_y = self.__get_whole_cycle_soc_x_y(
                self.train_discharge_cyc, self.x_indices, y_indices
            )
            test_raw_x, test_y = self.__get_whole_cycle_soc_x_y(
                self.test_discharge_cyc, self.x_indices, y_indices
            )

        train_scaled_x = self.__get_scaled_whole_cycle_x(
            train_raw_x, self.discharge_scalers
        )
        test_scaled_x = self.__get_scaled_whole_cycle_x(
            test_raw_x, self.discharge_scalers
        )

        train_x, test_x = self.__get_padded_whole_cycle(train_scaled_x, test_scaled_x)
        if not soh:
            train_y, test_y = self.__get_padded_whole_cycle(train_y, test_y)

        if multiple_output and soh:
            # (SOH only) duplicate the y values to multiple steps for each cycle
            train_y = np.repeat(train_y[:, None, :], train_x.shape[1], axis=1)
            test_y = np.repeat(test_y[:, None, :], test_x.shape[1], axis=1)

        self.logger.info(
            "Train x: %s, train y: %s | Test x: %s, test y: %s"
            % (train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        )

        return (train_x, train_y, test_x, test_y)

    # === ALL HELPER METHODS ARE NOW INCLUDED ===

    def __get_whole_cycle_soh_x_y(self, cyc, cap, x_indices, y_indices):
        x = list(map(lambda data: data[:, x_indices].astype("float32"), cyc))
        y = np.array(cap[:, y_indices], dtype="float32")
        return (x, y)

    def __get_whole_cycle_soc_x_y(self, cyc, x_indices, y_indices):
        x = np.array(list(map(lambda data: data[:, x_indices].astype("float32"), cyc)))
        y = np.array(list(map(lambda data: data[:, y_indices].astype("float32"), cyc)))
        return (x, y)

    def __get_scaled_whole_cycle_x(self, x, scalers):
        def map_func(data):
            result = []
            for i in range(len(scalers)):
                result.append(scalers[i].transform(data[:, [i]]).flatten())
            return np.array(result).T

        # Use dtype=object for lists of arrays of different lengths
        return np.array(list(map(map_func, x)), dtype=object)

    def __pad_cycle_list(self, cycle_list, required_step_count):
        """Pads a list of cycle arrays to the required step count."""

        def padding_map_func(data):
            # Ensure data is at least 2D for np.pad
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            pad_width = ((0, required_step_count - len(data)), (0, 0))
            return np.pad(data, pad_width, "constant", constant_values=0)

        # Handle empty list
        if not cycle_list or len(cycle_list) == 0:
            return np.array([])
        return np.array(list(map(padding_map_func, cycle_list)))

    def __get_padded_whole_cycle(self, train, test, min_cycle_length=None):
        all_cycles = []
        if train is not None and len(train) > 0:
            all_cycles.extend(train)
        if test is not None and len(test) > 0:
            all_cycles.extend(test)

        if not all_cycles:
            return (np.array([]), np.array([]))

        max_cycle_step_count = max(len(cycle) for cycle in all_cycles)
        required_step_count = max_cycle_step_count
        if min_cycle_length is not None:
            required_step_count = max(max_cycle_step_count, min_cycle_length)

        train_padded = self.__pad_cycle_list(train, required_step_count)
        test_padded = self.__pad_cycle_list(test, required_step_count)

        return (train_padded, test_padded)

    def get_discharge_single_step(self, output_capacity=False, soh=False):
        """x: [[voltage, current, temperature], ...]\n
        SOH y: [[soh/last_charging_capacity, remaining_time_to_cell_end], ...]
        SOC y: [[soc/discharging_capacity, remaining_time_to_cycle_end], ...]"""

        if soh:
            y_indices = [
                (
                    CapacityCols.CORRESPONDING_CHARGING_CAPACITY
                    if output_capacity
                    else CapacityCols.SOH
                ),
                CapacityCols.REMAINING_TIME_TO_CELL_END,
            ]
            train_x, train_y = self.__get_single_step_soh(
                self.train_discharge_cyc, self.train_discharge_cap, y_indices
            )
            test_x, test_y = self.__get_single_step_soh(
                self.test_discharge_cyc, self.test_discharge_cap, y_indices
            )
        else:
            y_indices = [
                CycleCols.DISCHARGING_CAPACITY if output_capacity else CycleCols.SOC,
                CycleCols.REMAINING_TIME_TO_CYCLE_END,
            ]
            train_x, train_y = self.__get_single_step_soc(
                self.train_discharge_cyc, y_indices
            )
            test_x, test_y = self.__get_single_step_soc(
                self.test_discharge_cyc, y_indices
            )

        for i in range(len(self.discharge_scalers)):
            train_x[:, [i]] = self.discharge_scalers[i].transform(train_x[:, [i]])
            test_x[:, [i]] = self.discharge_scalers[i].transform(test_x[:, [i]])

        self.logger.info(
            "Train x: %s, train y: %s | Test x: %s, test y: %s"
            % (train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        )

        return (train_x, train_y, test_x, test_y)

    def __get_single_step_soc(self, cyc, y_indices):
        concatenated_cyc = np.concatenate(cyc)
        x = concatenated_cyc[:, self.x_indices].astype("float32")
        y = concatenated_cyc[:, y_indices].astype("float32")
        return (x, y)

    def __get_single_step_soh(self, cyc, cap, y_indices):
        x = list()
        y = list()
        for i in range(len(cyc)):
            current_cyc = cyc[i]
            current_cap = cap[i]
            x.extend(current_cyc[:, self.x_indices])
            y.extend(np.repeat([current_cap[y_indices]], len(current_cyc), axis=0))
        x = np.array(x).astype("float32")
        y = np.array(y).astype("float32")
        return (x, y)

    def get_discharge_multiple_step(
        self, steps, output_capacity=False, multiple_output=False, soh=False
    ):
        """x: [[[voltage, current, temperature], ...] ...]\n
        SOH y (single step): [[soh/last_charging_capacity, remaining_time_to_cell_end], ...]\n
        SOH y (multiple steps): [[[soh/last_charging_capacity, remaining_time_to_cell_end], ...], ...]\n
        SOC y (single step): [[soc/discharging_capacity, remaining_time_to_cycle_end], ...]\n
        SOC y (multiple steps): [[[soc/discharging_capacity, remaining_time_to_cycle_end], ...], ...]\n
        (Pad zeros to begining steps)"""

        if soh:
            y_indices = [
                (
                    CapacityCols.CORRESPONDING_CHARGING_CAPACITY
                    if output_capacity
                    else CapacityCols.SOH
                ),
                CapacityCols.REMAINING_TIME_TO_CELL_END,
            ]
            train_x, train_y = self.__get_multiple_timesteps_soh(
                self.train_discharge_cyc,
                self.train_discharge_cap,
                y_indices,
                steps,
                multiple_output,
            )
            test_x, test_y = self.__get_multiple_timesteps_soh(
                self.test_discharge_cyc,
                self.test_discharge_cap,
                y_indices,
                steps,
                multiple_output,
            )
        else:
            y_indices = [
                CycleCols.DISCHARGING_CAPACITY if output_capacity else CycleCols.SOC,
                CycleCols.REMAINING_TIME_TO_CYCLE_END,
            ]
            train_x, train_y = self.__get_multiple_timesteps_soc(
                self.train_discharge_cyc, y_indices, steps, multiple_output
            )
            test_x, test_y = self.__get_multiple_timesteps_soc(
                self.test_discharge_cyc, y_indices, steps, multiple_output
            )

        train_x = self.__scale_multiple_timestep(train_x)
        test_x = self.__scale_multiple_timestep(test_x)

        self.logger.info(
            "Train x: %s, train y: %s | Test x: %s, test y: %s"
            % (train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        )

        return (train_x, train_y, test_x, test_y)

    def __get_multiple_timesteps_soc(self, cyc, y_indices, steps, multiple_output):
        all_x, all_y = [], []
        for cycle in cyc:
            x, y = self.__cycle_to_multiple_steps_soc(
                y_indices, steps, multiple_output, cycle
            )
            all_x.append(np.array(x))
            all_y.append(np.array(y))
        all_x = np.concatenate(np.array(all_x)).astype("float32")
        all_y = np.concatenate(np.array(all_y)).astype("float32")
        return all_x, all_y

    def __cycle_to_multiple_steps_soc(
        self, y_indices, steps, multiple_output, cycle, x_indices=None
    ):
        if x_indices is None:
            x_indices = self.x_indices
        x, y = [], []
        for i in range(cycle.shape[0]):
            start_ix = i - steps + 1
            x_seq, y_seq = [], []
            y_seq = cycle[i, y_indices]
            # start index is negative, pad zeros to the sequence
            if start_ix < 0:
                x_seq = np.zeros((abs(start_ix), len(x_indices)))
                x_seq = np.append(x_seq, cycle[0 : i + 1, x_indices], axis=0)
                if multiple_output:
                    y_seq = np.zeros((abs(start_ix), len(y_indices)))
                    y_seq = np.append(y_seq, cycle[0 : i + 1, y_indices], axis=0)
            else:
                x_seq = cycle[start_ix : i + 1, x_indices]
                if multiple_output:
                    y_seq = cycle[start_ix : i + 1, y_indices]
            x.append(x_seq)
            y.append(y_seq)
        return x, y

    def __get_multiple_timesteps_soh(self, cyc, cap, y_indices, steps, multiple_output):
        all_x, all_y = [], []
        for i in range(len(cyc)):
            cycle = cyc[i]
            x, y = [], []
            for j in range(len(cycle)):
                start_ix = j - steps + 1
                x_seq, y_seq = [], []
                y_seq = cap[i, y_indices]
                # start index is negative, pad zeros to the sequence
                if start_ix < 0:
                    x_seq = np.zeros((abs(start_ix), len(self.x_indices)))
                    x_seq = np.append(x_seq, cycle[0 : j + 1, self.x_indices], axis=0)
                else:
                    x_seq = cycle[start_ix : j + 1, self.x_indices]

                if multiple_output:
                    y_seq = np.repeat([y_seq], steps, axis=0)

                x.append(x_seq)
                y.append(y_seq)
            all_x.append(np.array(x))
            all_y.append(np.array(y))
        all_x = np.concatenate(np.array(all_x)).astype("float32")
        all_y = np.concatenate(np.array(all_y)).astype("float32")
        return all_x, all_y

    def __scale_multiple_timestep(self, x):
        for i in range(len(self.discharge_scalers)):
            # Reshape to 2d array for scaling and reshape back to 3d
            two_d_x = x[:, :, [i]].reshape(x.shape[0] * x.shape[1], 1)
            scaled_two_d_x = self.discharge_scalers[i].transform(two_d_x)
            x[:, :, [i]] = scaled_two_d_x.reshape((x.shape[0], x.shape[1], 1))
        return x

    def get_discharge_grouped_multiple_steps(
        self, steps, output_capacity=False, multiple_output=False
    ):
        train_x, train_y, test_x, test_y = self.get_discharge_whole_cycle(
            output_capacity=output_capacity,
            # always get multiple output form whole cycle, so that we can split them to multiple steps
            multiple_output=True,
            soh=False,
        )

        self.logger.info("Spliting whole cycles to multiple steps...")

        train_x, train_y = self.__whole_cycle_to_multiple_step(
            steps, multiple_output, train_x, train_y
        )
        test_x, test_y = self.__whole_cycle_to_multiple_step(
            steps, multiple_output, test_x, test_y
        )

        self.logger.info(
            "Train x: %s, train y: %s | Test x: %s, test y: %s"
            % (train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        )

        return (train_x, train_y, test_x, test_y)

    def __whole_cycle_to_multiple_step(
        self, steps, multiple_output, whole_cycle_x, whole_cycle_y
    ):
        x_indices = np.arange(whole_cycle_x.shape[-1])
        y_indices = np.arange(whole_cycle_y.shape[-1])
        new_x = []
        new_y = []
        for i in range(len(whole_cycle_x)):
            x, y = self.__cycle_to_multiple_steps(
                y_indices,
                steps,
                multiple_output,
                whole_cycle_x[i],
                whole_cycle_y[i],
                x_indices,
            )
            new_x.append(x)
            new_y.append(y)
        whole_cycle_x = np.array(new_x)
        whole_cycle_y = np.array(new_y)
        return whole_cycle_x, whole_cycle_y

    def __cycle_to_multiple_steps(
        self, y_indices, steps, multiple_output, cycle_x, cycle_y, x_indices=None
    ):
        if x_indices is None:
            x_indices = self.x_indices
        x, y = [], []
        for i in range(cycle_x.shape[0]):
            start_ix = i - steps + 1
            x_seq, y_seq = [], []
            y_seq = cycle_y[i, y_indices]
            # start index is negative, pad zeros to the sequence
            if start_ix < 0:
                x_seq = np.zeros((abs(start_ix), len(x_indices)))
                x_seq = np.append(x_seq, cycle_x[0 : i + 1, x_indices], axis=0)
                if multiple_output:
                    y_seq = np.zeros((abs(start_ix), len(y_indices)))
                    y_seq = np.append(y_seq, cycle_y[0 : i + 1, y_indices], axis=0)
            else:
                x_seq = cycle_x[start_ix : i + 1, x_indices]
                if multiple_output:
                    y_seq = cycle_y[start_ix : i + 1, y_indices]
            x.append(x_seq)
            y.append(y_seq)
        return x, y

    def keep_only_capacity(
        self, y, is_multiple_output=False, is_grouped_multiple_step=False
    ):
        if is_grouped_multiple_step:
            if is_multiple_output:
                new_y = y[:, :, :, 0]
            else:
                new_y = y[:, :, 0]
        else:
            if is_multiple_output:
                new_y = y[:, :, 0]
            else:
                new_y = y[:, 0]
        self.logger.info("New y: %s" % (new_y.shape,))
        return new_y

    def keep_only_time(
        self, y, is_multiple_output=False, is_grouped_multiple_step=False
    ):
        if is_grouped_multiple_step:
            if is_multiple_output:
                new_y = y[:, :, :, 1]
            else:
                new_y = y[:, :, 1]
        else:
            if is_multiple_output:
                new_y = y[:, :, 1]
            else:
                new_y = y[:, 1]
        self.logger.info("New y: %s" % (new_y.shape,))
        return new_y

    # === REFACTORED get_discharge_whole_cycle_future ===
    def get_discharge_whole_cycle_future(
        self, train_names, test_names, val_names=None, min_cycle_length=None
    ):
        y_indices = [CapacityCols.CORRESPONDING_CHARGING_CAPACITY]

        # 1. Group raw data into a dictionary to loop over
        raw_data_all = {
            "train": {
                "cyc": self.train_discharge_cyc,
                "cap": self.train_discharge_cap,
                "names": train_names,
            },
            "test": {
                "cyc": self.test_discharge_cyc,
                "cap": self.test_discharge_cap,
                "names": test_names,
            },
        }

        if val_names and self.val_discharge_cyc is not None:
            raw_data_all["val"] = {
                "cyc": self.val_discharge_cyc,
                "cap": self.val_discharge_cap,
                "names": val_names,
            }

        processed_data = {}
        all_raw_x = []
        all_time_raw = []

        # 2. Process all splits in a loop (TRAIN, TEST, VAL)
        for split_name, data in raw_data_all.items():
            if data["cyc"] is None or len(data["cyc"]) == 0:
                self.logger.warning(f"No data found for {split_name} split. Skipping.")
                continue

            # Get main X and Y (SOH)
            raw_x, y = self.__get_whole_cycle_soh_x_y(
                data["cyc"], data["cap"], self.x_indices, y_indices
            )

            # Get time data
            time_raw, _ = self.__get_whole_cycle_soh_x_y(
                data["cyc"], data["cap"], [CycleCols.STEP_TIME], y_indices
            )

            # Get initial lengths
            initial_lengths = [len(i) for i in raw_x]

            # Get battery range
            battery_name_cycle = data["cap"][:, [CapacityCols.TEST_NAME]]
            battery_range = np.array(
                [np.where(battery_name_cycle == x)[0][-1] + 1 for x in data["names"]]
            )

            # Store processed data and lists for padding
            processed_data[split_name] = {
                "raw_x": raw_x,
                "y": y,
                "time_raw": time_raw,
                "lengths": initial_lengths,
                "range": battery_range,
            }

            all_raw_x.extend(raw_x)
            all_time_raw.extend(time_raw)

        # 3. Find max padding length across *all* splits
        if not all_raw_x:
            self.logger.error("No data processed. Returning empty tuples.")
            return (None,) * 18

        max_len = max(len(cycle) for cycle in all_raw_x)
        if min_cycle_length:
            max_len = max(max_len, min_cycle_length)

        # 4. Pad, post-process, and store final data
        final_data = {}
        for split_name, data in processed_data.items():
            x_padded = self.__pad_cycle_list(data["raw_x"], max_len)
            time_padded = self.__pad_cycle_list(data["time_raw"], max_len)

            # Post-process current (negative)
            x_padded[:, :, 1] = np.negative(x_padded[:, :, 1])
            current = x_padded[:, :, 1]

            final_data[split_name] = {
                "x": x_padded,
                "y": data["y"],
                "time": time_padded,
                "current": current,
                "lengths": data["lengths"],
                "range": data["range"],
            }

        # 5. Log info for all available splits
        train_log = f"Train x: {final_data['train']['x'].shape}, train y soh: {final_data['train']['y'].shape}"
        test_log = f"Test x: {final_data['test']['x'].shape}, test y soh: {final_data['test']['y'].shape}"
        val_log = ""
        if "val" in final_data:
            val_log = f"Validation x: {final_data['val']['x'].shape}, validation y soh: {final_data['val']['y'].shape}"
        self.logger.info(f"{train_log} | {val_log} | {test_log}")

        # 6. Unpack all 18 variables and return
        train_data = final_data.get(
            "train",
            {
                "x": None,
                "y": None,
                "range": None,
                "time": None,
                "current": None,
                "lengths": None,
            },
        )
        test_data = final_data.get(
            "test",
            {
                "x": None,
                "y": None,
                "range": None,
                "time": None,
                "current": None,
                "lengths": None,
            },
        )
        # Provide Nones if validation data was missing
        val_data = final_data.get(
            "val",
            {
                "x": None,
                "y": None,
                "range": None,
                "time": None,
                "current": None,
                "lengths": None,
            },
        )

        return (
            train_data["x"],
            train_data["y"],
            val_data["x"],
            val_data["y"],
            test_data["x"],
            test_data["y"],
            train_data["range"],
            val_data["range"],
            test_data["range"],
            train_data["time"],
            val_data["time"],
            test_data["time"],
            train_data["current"],
            val_data["current"],
            test_data["current"],
            train_data["lengths"],
            val_data["lengths"],
            test_data["lengths"],
        )
