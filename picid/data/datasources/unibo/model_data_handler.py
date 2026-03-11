import logging

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .unibo_powertools_data import CapacityCols, CycleCols


class ModelDataHandler:
    """
    This class is the data processor for the UNIBO dataset.

    It takes the raw data loaded by `UniboPowertoolsData` and transforms it
    into various formats (e.g., padded cycles, flattened timesteps,
    sliding windows) required by different types of machine learning models.

    The key method for your RUL pipeline is `get_discharge_whole_cycle_future`.
    """

    def __init__(self, dataset, x_indices, scaler_type=MinMaxScaler):
        """
        Initializes the data handler.

        Parameters
        ----------
        dataset
            An instantiated `UniboPowertoolsData` object that has already
            called its `prepare_data` method.
        x_indices
            A list of column indices from the raw cycle data to be
            used as features (e.g., [CycleCols.VOLTAGE, ...]).
        scaler_type
            The scaler class to use (e.g., MinMaxScaler).
        """
        self.logger = logging.getLogger()
        self.dataset = dataset
        self.x_indices = x_indices
        self.scaler_type = scaler_type

        # === REFACTORED __init__ ===

        # 1. Get the data dictionaries from the dataset.
        #    (We refactored UniboPowertoolsData to return dicts keyed by 'train', 'val', 'test')
        (self.charge_cyc_all, self.charge_cap_all) = self.dataset.get_charge_data()
        (self.discharge_cyc_all, self.discharge_cap_all) = (
            self.dataset.get_discharge_data()
        )

        # 2. Assign attributes for backward compatibility.
        #    Many "legacy" methods in this class (like `get_discharge_single_step`)
        #    were written *before* the refactor and expect 'self.train_discharge_cyc',
        #    not the new dictionary `self.discharge_cyc_all['train']`.
        #    These lines allow those old methods to keep working without modification.
        self.train_charge_cyc = self.charge_cyc_all.get("train")
        self.train_charge_cap = self.charge_cap_all.get("train")
        self.test_charge_cyc = self.charge_cyc_all.get("test")
        self.test_charge_cap = self.charge_cap_all.get("test")

        self.train_discharge_cyc = self.discharge_cyc_all.get("train")
        self.train_discharge_cap = self.discharge_cap_all.get("train")
        self.test_discharge_cyc = self.discharge_cyc_all.get("test")
        self.test_discharge_cap = self.discharge_cap_all.get("test")

        # 3. Also assign validation data.
        #    This is used by the new refactored `get_discharge_whole_cycle_future` method.
        self.val_discharge_cyc = self.discharge_cyc_all.get("val")
        self.val_discharge_cap = self.discharge_cap_all.get("val")

        # 4. Fit scalers to the training data.
        #    This call relies on `self.train_charge_cyc` and `self.train_discharge_cyc`
        #    which we just assigned in step 2.
        # self.__assign_scalers()

    def __assign_scalers(self):
        """Fits scalers based on the training data for both charge and discharge."""
        self.charge_scalers = self.__create_scalers(self.train_charge_cyc)
        self.discharge_scalers = self.__create_scalers(self.train_discharge_cyc)

    def __create_scalers(self, cyc):
        """Helper to create a list of scalers, one for each feature index."""
        scalers = []
        for index in self.x_indices:
            scalers.append(self.__create_scaler(cyc, index))
        return scalers

    def __create_scaler(self, cyc, col_index):
        """Fits a single scaler to a single feature column from the training data."""
        # Concatenate all training cycles into one giant column
        data = np.concatenate(cyc)[:, col_index].reshape(-1, 1)
        scaler_x = self.scaler_type()
        # Fit the scaler. The scaler object is mutated and stored.
        scaler_x.fit_transform(data)
        return scaler_x

    def get_scalers(self):
        """Public method to retrieve the fitted scalers."""
        return self.charge_scalers, self.discharge_scalers

    def get_discharge_whole_cycle(
        self, output_capacity=False, multiple_output=False, soh=False
    ):
        """
        (Legacy Method)
        Returns data as padded, whole cycles. This method formats data for
        predicting either SOH (a single value per cycle) or
        SOC (a time-series value per timestep).
        """

        if soh:
            # --- SOH Prediction ---
            # SOH is one value per cycle, extracted from the 'capacity' arrays.
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
            # --- SOC Prediction ---
            # SOC is a value for every timestep, extracted from the 'cycle' arrays.
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

        # Scale the features (X)
        train_scaled_x = self.__get_scaled_whole_cycle_x(
            train_raw_x, self.discharge_scalers
        )
        test_scaled_x = self.__get_scaled_whole_cycle_x(
            test_raw_x, self.discharge_scalers
        )

        # Pad all X arrays to the same length (that of the longest cycle)
        train_x, test_x = self.__get_padded_whole_cycle(train_scaled_x, test_scaled_x)
        if not soh:
            # If target is SOC (time-series), it must also be padded.
            train_y, test_y = self.__get_padded_whole_cycle(train_y, test_y)

        if multiple_output and soh:
            # For SOH models, sometimes we want to predict SOH at every timestep.
            # This repeats the single SOH value for all timesteps in the cycle.
            train_y = np.repeat(train_y[:, None, :], train_x.shape[1], axis=1)
            test_y = np.repeat(test_y[:, None, :], test_x.shape[1], axis=1)

        self.logger.info(
            "Train x: %s, train y: %s | Test x: %s, test y: %s"
            % (train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        )

        return (train_x, train_y, test_x, test_y)

    # === HELPER METHODS for legacy getters ===

    def __get_whole_cycle_soh_x_y(self, cyc, cap, x_indices, y_indices):
        """Helper to extract X (list of arrays) and Y (single SOH value per cycle)."""
        # X is a list of arrays because each cycle has a different length (ragged)
        x = list(map(lambda data: data[:, x_indices].astype("float32"), cyc))
        # Y is a simple 2D array, as it's one SOH value per cycle (uniform)
        y = np.array(cap[:, y_indices], dtype="float32")
        return (x, y)

    def __get_whole_cycle_soc_x_y(self, cyc, x_indices, y_indices):
        """Helper to extract X and Y (SOC, a value for each timestep)."""
        # Both X and Y are lists of arrays (ragged)
        x = list(map(lambda data: data[:, x_indices].astype("float32"), cyc))
        y = list(map(lambda data: data[:, y_indices].astype("float32"), cyc))
        return (x, y)

    def __get_scaled_whole_cycle_x(self, x, scalers):
        """Applies the fitted scalers to a list of unpadded cycle arrays."""

        def map_func(data):
            result = []
            for i in range(len(scalers)):
                # Scale each feature column (V, I, T) individually
                result.append(scalers[i].transform(data[:, [i]]).flatten())
            return np.array(result).T

        # The result is a list of scaled arrays.
        # np.array(..., dtype=object) is used to handle lists of arrays of different lengths.
        return np.array(list(map(map_func, x)), dtype=object)

    def __pad_cycle_list(self, cycle_list, required_step_count):
        """Helper function: Pads a list of cycle arrays to a fixed step count."""

        def padding_map_func(data):
            # Ensure data is 2D for np.pad
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            # Pad with 0s at the end of the sequence (axis 0)
            pad_width = ((0, required_step_count - len(data)), (0, 0))
            return np.pad(data, pad_width, "constant", constant_values=0)

        # Handle empty list
        if not cycle_list or len(cycle_list) == 0:
            return np.array([])
        # Apply the padding function to every cycle in the list
        return np.array(list(map(padding_map_func, cycle_list)))

    def __get_padded_whole_cycle(self, train, test, min_cycle_length=None):
        """
        Finds the longest cycle across both train and test sets,
        and pads all cycles to that single length.
        """
        all_cycles = []
        if train is not None and len(train) > 0:
            all_cycles.extend(train)
        if test is not None and len(test) > 0:
            all_cycles.extend(test)

        if not all_cycles:
            return (np.array([]), np.array([]))

        # Find the global maximum cycle length
        max_cycle_step_count = max(len(cycle) for cycle in all_cycles)
        required_step_count = max_cycle_step_count
        if min_cycle_length is not None:
            required_step_count = max(max_cycle_step_count, min_cycle_length)

        # Pad both sets using the same global max length
        # Ensure list type so __pad_cycle_list empty check is well-defined (no array truth value)
        train_list = list(train) if isinstance(train, np.ndarray) else train
        test_list = list(test) if isinstance(test, np.ndarray) else test
        train_padded = self.__pad_cycle_list(train_list, required_step_count)
        test_padded = self.__pad_cycle_list(test_list, required_step_count)

        return (train_padded, test_padded)

    def get_discharge_single_step(self, output_capacity=False, soh=False):
        """
        (Legacy Method)
        Returns data as a single, large 2D array.
        All cycles are concatenated together, flattening the data.
        Shape: [total_timesteps, n_features]
        Good for simple Feed-Forward Networks or similar models.
        """

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

        # Scale the giant 2D arrays
        for i in range(len(self.discharge_scalers)):
            train_x[:, [i]] = self.discharge_scalers[i].transform(train_x[:, [i]])
            test_x[:, [i]] = self.discharge_scalers[i].transform(test_x[:, [i]])

        self.logger.info(
            "Train x: %s, train y: %s | Test x: %s, test y: %s"
            % (train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        )

        return (train_x, train_y, test_x, test_y)

    def __get_single_step_soc(self, cyc, y_indices):
        """Helper to flatten SOC data."""
        # Simply concatenate all cycle arrays into one long array
        concatenated_cyc = np.concatenate(cyc)
        x = concatenated_cyc[:, self.x_indices].astype("float32")
        y = concatenated_cyc[:, y_indices].astype("float32")
        return (x, y)

    def __get_single_step_soh(self, cyc, cap, y_indices):
        """Helper to flatten SOH data. This is trickier."""
        x = list()
        y = list()
        # For each cycle...
        for i in range(len(cyc)):
            current_cyc = cyc[i]  # The feature array [timesteps, n_features]
            current_cap = cap[i]  # The single SOH value [n_y_indices]
            # Add all features to the list
            x.extend(current_cyc[:, self.x_indices])
            # Repeat the single SOH value for every timestep in that cycle
            y.extend(np.repeat([current_cap[y_indices]], len(current_cyc), axis=0))
        x = np.array(x).astype("float32")
        y = np.array(y).astype("float32")
        return (x, y)

    def get_discharge_multiple_step(
        self, steps, output_capacity=False, multiple_output=False, soh=False
    ):
        """
        (Legacy Method)
        Returns data as a 3D array created using a sliding window of length 'steps'.
        Shape: [n_sequences, 'steps', n_features]
        Good for LSTMs, GRUs, or 1D-CNNs.
        """

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

        # Scaling a 3D array requires reshaping
        train_x = self.__scale_multiple_timestep(train_x)
        test_x = self.__scale_multiple_timestep(test_x)

        self.logger.info(
            "Train x: %s, train y: %s | Test x: %s, test y: %s"
            % (train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        )

        return (train_x, train_y, test_x, test_y)

    def __get_multiple_timesteps_soc(self, cyc, y_indices, steps, multiple_output):
        """Helper to create sliding windows for SOC data."""
        all_x, all_y = [], []
        for cycle in cyc:
            # Create sliding windows for one cycle
            x, y = self.__cycle_to_multiple_steps_soc(
                y_indices, steps, multiple_output, cycle
            )
            all_x.append(np.array(x))
            all_y.append(np.array(y))
        # Concatenate windows from all cycles
        all_x = np.concatenate(all_x).astype("float32")
        all_y = np.concatenate(all_y).astype("float32")
        return all_x, all_y

    def __cycle_to_multiple_steps_soc(
        self, y_indices, steps, multiple_output, cycle, x_indices=None
    ):
        """Helper: applies a sliding window to a single cycle."""
        if x_indices is None:
            x_indices = self.x_indices
        x, y = [], []
        # For each timestep 'i' in the cycle...
        for i in range(cycle.shape[0]):
            # 'i' is the end of the window
            start_ix = i - steps + 1  # 'start_ix' is the beginning
            x_seq, y_seq = [], []
            y_seq = cycle[i, y_indices]  # Target is the value at the end 'i'

            # If the window starts before the cycle (e.g., at step 2 with window=10),
            # pad the beginning with zeros.
            if start_ix < 0:
                x_seq = np.zeros((abs(start_ix), len(x_indices)))  # Pad with zeros
                x_seq = np.append(
                    x_seq, cycle[0 : i + 1, x_indices], axis=0
                )  # Add real data
                if multiple_output:
                    y_seq = np.zeros((abs(start_ix), len(y_indices)))
                    y_seq = np.append(y_seq, cycle[0 : i + 1, y_indices], axis=0)
            else:
                # Normal case: just slice the window
                x_seq = cycle[start_ix : i + 1, x_indices]
                if multiple_output:
                    y_seq = cycle[start_ix : i + 1, y_indices]
            x.append(x_seq)
            y.append(y_seq)
        return x, y

    def __get_multiple_timesteps_soh(self, cyc, cap, y_indices, steps, multiple_output):
        """Helper to create sliding windows for SOH data."""
        all_x, all_y = [], []
        for i in range(len(cyc)):  # For each cycle 'i'
            cycle = cyc[i]
            x, y = [], []
            for j in range(len(cycle)):  # For each timestep 'j' in that cycle
                start_ix = j - steps + 1
                x_seq, y_seq = [], []
                # The SOH target is the same (cap[i]) for all timesteps 'j'
                y_seq = cap[i, y_indices]

                # Pad with zeros if at the beginning
                if start_ix < 0:
                    x_seq = np.zeros((abs(start_ix), len(self.x_indices)))
                    x_seq = np.append(x_seq, cycle[0 : j + 1, self.x_indices], axis=0)
                else:
                    x_seq = cycle[start_ix : j + 1, self.x_indices]

                if multiple_output:
                    # Repeat the single SOH value for all 'steps'
                    y_seq = np.repeat([y_seq], steps, axis=0)

                x.append(x_seq)
                y.append(y_seq)
            all_x.append(np.array(x))
            all_y.append(np.array(y))
        all_x = np.concatenate(all_x).astype("float32")
        all_y = np.concatenate(all_y).astype("float32")
        return all_x, all_y

    def __scale_multiple_timestep(self, x):
        """Helper to scale a 3D (sliding window) array."""
        for i in range(len(self.discharge_scalers)):
            # To scale a 3D array [samples, steps, features], we must reshape
            # to 2D [samples*steps, features]
            two_d_x = x[:, :, [i]].reshape(x.shape[0] * x.shape[1], 1)
            scaled_two_d_x = self.discharge_scalers[i].transform(two_d_x)
            # Reshape back to 3D
            x[:, :, [i]] = scaled_two_d_x.reshape((x.shape[0], x.shape[1], 1))
        return x

    def get_discharge_grouped_multiple_steps(
        self, steps, output_capacity=False, multiple_output=False
    ):
        """(Legacy Method) Another variant for creating sequence data."""
        train_x, train_y, test_x, test_y = self.get_discharge_whole_cycle(
            output_capacity=output_capacity,
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
        """Helper for get_discharge_grouped_multiple_steps"""
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
        """Helper for get_discharge_grouped_multiple_steps"""
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
        """Helper to select just the first column of a 2-column Y target."""
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
        """Helper to select just the second column of a 2-column Y target."""
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

    # =========================================================================
    # === THE METHOD YOU ARE USING ===
    # =========================================================================
    def get_discharge_whole_cycle_future(
        self, train_names, test_names, val_names=None, min_cycle_length=None
    ):
        """
        This is the new, refactored method used by your 'read_data' function.
        It is the only method that correctly handles the 'validation' split.

        Its sole purpose is to gather all the necessary arrays (x, y_soh, current, time)
        and pad them to a uniform length across ALL splits (train, val, test).
        This data is then passed *out* to the 'RulHandler' in your 'read_data' function,
        which will perform the final RUL calculation.
        """

        # We only need the raw SOH (capacity) value to find the EoL threshold.
        # This is the `y_soh` that `read_data` receives.
        y_indices = [CapacityCols.CORRESPONDING_CHARGING_CAPACITY]

        # 1. Group all raw data into a dictionary for easy looping.
        #    This uses the data loaded in the `__init__` method.
        raw_data_all = {
            "train": {
                "cyc": self.train_discharge_cyc,  # DISCHARGE data
                "cap": self.train_discharge_cap,
                "names": train_names,
            },
            "test": {
                "cyc": self.test_discharge_cyc,  # DISCHARGE data
                "cap": self.test_discharge_cap,
                "names": test_names,
            },
        }

        # Add validation data to the loop if it exists
        if val_names and self.val_discharge_cyc is not None:
            raw_data_all["val"] = {
                "cyc": self.val_discharge_cyc,  # DISCHARGE data
                "cap": self.val_discharge_cap,
                "names": val_names,
            }

        processed_data = {}  # To store intermediate, unpadded results
        all_raw_x = []  # A list to find the longest cycle for padding
        all_time_raw = []  # A list to find the longest cycle for padding

        # 2. Process all splits in a loop (TRAIN, TEST, VAL)
        for split_name, data in raw_data_all.items():
            if data["cyc"] is None or len(data["cyc"]) == 0:
                self.logger.warning(f"No data found for {split_name} split. Skipping.")
                continue

            # Get main X (V, I, T) and Y (Capacity/SOH)
            # `raw_x` is a list of unpadded arrays. `y` is a 2D array of SOH values.
            raw_x, y = self.__get_whole_cycle_soh_x_y(
                data["cyc"], data["cap"], self.x_indices, y_indices
            )

            # Get the 'time' column in the same (unpadded) format
            time_raw, _ = self.__get_whole_cycle_soh_x_y(
                data["cyc"], data["cap"], [CycleCols.STEP_TIME], y_indices
            )

            # Get the original, unpadded lengths (used later for unpadding)
            initial_lengths = [len(i) for i in raw_x]

            # Get battery range (used later for splitting by battery)
            battery_name_cycle = data["cap"][:, [CapacityCols.TEST_NAME]]
            battery_range = np.array(
                [np.where(battery_name_cycle == x)[0][-1] + 1 for x in data["names"]]
            )

            # Store all unpadded data temporarily
            processed_data[split_name] = {
                "raw_x": raw_x,
                "y": y,
                "time_raw": time_raw,
                "lengths": initial_lengths,
                "range": battery_range,
            }

            # Add to global lists to find the max padding length
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
            # Pad X features and Time features to the same global max_len
            x_padded = self.__pad_cycle_list(data["raw_x"], max_len)
            time_padded = self.__pad_cycle_list(data["time_raw"], max_len)

            # ==================================================================
            # === THIS IS THE KEY TO YOUR QUESTION ===
            # ==================================================================
            # In the UNIBO dataset, discharge current is NEGATIVE.
            # The `RulHandler` (from the other file) integrates `current > 0`.
            # To make these two pieces of code work, we *must* flip the sign
            # of the discharge current, making it POSITIVE.
            #
            # x_padded[:, :, 1] is the current column (index 1)
            x_padded[:, :, 1] = np.negative(x_padded[:, :, 1])

            # This `current` variable is now a POSITIVE discharge current.
            # This is what gets passed to the RulHandler, and this is why
            # the RulHandler's `current > 0` logic correctly integrates
            # the DISCHARGE throughput.
            current = x_padded[:, :, 1]
            # ==================================================================

            # Store all the final, padded arrays
            final_data[split_name] = {
                "x": x_padded,
                "y": data["y"],
                "time": time_padded,
                "current": current,  # The positive-flipped discharge current
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

        # 6. Unpack all 18 variables and return in the expected order
        #    for the 'read_data' function.
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
