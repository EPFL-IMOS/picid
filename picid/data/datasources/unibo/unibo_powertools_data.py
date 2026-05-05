import logging
from pathlib import Path
from phmd.download import download
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TEST_RESULT_DATA_PATH = (
    "data/unibo-powertools-dataset/unibo-powertools-dataset/test_result.csv"
)
TEST_RESULT_TRIAL_END_DATA_PATH = (
    "data/unibo-powertools-dataset/unibo-powertools-dataset/test_result_trial_end.csv"
)

# (test_name, record_id)
ABNORMAL_CYCLE_RECORDS = [
    ("006-EE-2.85-0820-S", 621391),  # Discharge capacity dropped abnormally
    ("006-EE-2.85-0820-S", 621392),  # Discharge capacity dropped abnormally
]
ABNORMAL_CAPACITY_RECRODS = [
    ("007-EE-2.85-0820-S", 623002)  # Not the last row in cycle
]


class CycleCols:
    TEST_NAME = 0
    RECORD_ID = 1
    TIME = 2
    STEP_TIME = 3
    LINE = 4
    VOLTAGE = 5
    CURRENT = 6
    CHARGING_CAPACITY = 7
    DISCHARGING_CAPACITY = 8
    WH_CHARGING = 9
    WH_DISCHARGING = 10
    TEMPERATURE = 11
    CYCLE_COUNT = 12
    SOC = 13
    REMAINING_TIME_TO_CYCLE_END = 14


class CapacityCols:
    TEST_NAME = 0
    RECORD_ID = 1
    TIME = 2
    STEP_TIME = 3
    LINE = 4
    VOLTAGE = 5
    CURRENT = 6
    CHARGING_CAPACITY = 7
    DISCHARGING_CAPACITY = 8
    WH_CHARGING = 9
    WH_DISCHARGING = 10
    TEMPERATURE = 11
    CYCLE_COUNT = 12
    MAX_TEMPERATURE = 13
    AVERAGE_TENSION = 14
    REMAINING_TIME_TO_CELL_END = 15
    MAXIMUM_CAPACITY = 16
    NOMINAL_CAPACITY = 17
    SOH = 18
    CORRESPONDING_CHARGING_CAPACITY = 19


class UniboPowertoolsData:
    def __init__(
        self,
        test_types=[],
        chunk_size=1000000,
        lines=[37, 40],
        charge_line=37,
        discharge_line=40,
        base_path=None,
        force_download=False,
    ):
        self.test_types = test_types if test_types is not None else []
        self.chunksize = chunk_size
        self.lines = lines if lines is not None else []
        self.charge_line = charge_line
        self.discharge_line = discharge_line
        self.base_path = Path(base_path).expanduser()

        # check if self.data_path exist
        download("UNIBO21", cache_dir=self.base_path, force=force_download, unzip=True)
        self.cyc_path = self.base_path / "datasets/UNIBO21/data/test_result.csv"
        self.cap_path = (
            self.base_path / "datasets/UNIBO21/data/test_result_trial_end.csv"
        )

        self.__load_raw_data()

    def __load_raw_data(self):
        self.__load_csv_to_raw()
        self.__clean_cycle_raw()
        self.__clean_capacity_raw()
        self.__assign_charge_raw()
        self.__assign_discharge_raw()

    def __load_csv_to_raw(self):
        logger.debug(
            "Start loading data with lines: %s, types: %s and chunksize: %s..."
            % (self.lines, self.test_types, self.chunksize)
        )

        iter_cyc = pd.read_csv(self.cyc_path, chunksize=self.chunksize, iterator=True)
        self.cycle_raw = pd.concat(self.__filter_raw_chunk(iter_cyc))

        iter_cap = pd.read_csv(self.cap_path, chunksize=self.chunksize, iterator=True)
        self.cap_raw = pd.concat(self.__filter_raw_chunk(iter_cap))

        logger.debug("Finish loading data.")
        logger.info(
            "Loaded raw dataset A data with cycle row count: %s and capacity row count: %s"
            % (len(self.cycle_raw), len(self.cap_raw))
        )

    def __filter_raw_chunk(self, iter_chunk):
        # Collect all conditions first.
        # If no test name and lines specified, get all data from chunk without filtering
        conditions = list()
        if len(self.test_types) > 0:
            conditions.append("test_name.str.endswith(tuple(%s))" % self.test_types)
        if len(self.lines) > 0:
            conditions.append("line.isin(%s)" % self.lines)

        filter_cks = []
        for chunk in iter_chunk:
            if len(conditions) > 0:
                chunk = chunk.query("&".join(conditions), engine="python")
            filter_cks.append(chunk)
        return filter_cks

    def __clean_cycle_raw(self):
        logger.debug("Start cleaning cycle raw data...")
        count_before = len(self.cycle_raw)

        # Voltage outside 0.1 ~ 5.0 are seen as abnormal dataset
        self.cycle_raw = self.cycle_raw.drop(
            self.cycle_raw[
                (self.cycle_raw["voltage"] > 5.0) | (self.cycle_raw["voltage"] < 0.1)
            ].index
        )

        # Filter all predefined abnormal records
        self.cycle_raw = self.__filter_predefined(
            self.cycle_raw, ABNORMAL_CYCLE_RECORDS
        )

        logger.debug("Finish cleaning cycle raw data.")
        logger.info(
            "Removed %s rows of abnormal cycle raw data."
            % (count_before - len(self.cycle_raw))
        )

    def __filter_predefined(self, raw_data, predefined_records):
        for abn_record in predefined_records:
            raw_data.drop(
                raw_data[
                    (raw_data["test_name"] == abn_record[0])
                    & (raw_data["record_id"] == abn_record[1])
                ].index,
                inplace=True,
            )
        return raw_data

    def __clean_capacity_raw(self):
        logger.debug("Start cleaning capacity raw data...")
        count_before = len(self.cap_raw)

        # Filter all predefined abnormal records
        self.cap_raw = self.__filter_predefined(self.cap_raw, ABNORMAL_CAPACITY_RECRODS)

        logger.debug("Finish cleaning capacity raw data.")
        logger.info(
            "Removed %s rows of abnormal capacity raw data."
            % (count_before - len(self.cap_raw))
        )

    def __assign_charge_raw(self):
        logger.debug("Start assigning charging raw data...")

        self.charge_cyc_raw = self.cycle_raw[self.cycle_raw["line"] == self.charge_line]
        self.charge_cap_raw = self.cap_raw[self.cap_raw["line"] == self.charge_line]

        logger.debug("Finish assigning charging raw data.")
        logger.info(
            "[Charging] cycle raw count: %s, capacity raw count: %s"
            % (len(self.charge_cyc_raw), len(self.charge_cap_raw))
        )

    def __assign_discharge_raw(self):
        logger.debug("Start assigning discharging raw data...")

        self.discharge_cyc_raw = self.cycle_raw[
            self.cycle_raw["line"] == self.discharge_line
        ]
        self.discharge_cap_raw = self.cap_raw[
            self.cap_raw["line"] == self.discharge_line
        ]

        logger.debug("Finish assigning discharging raw data.")
        logger.info(
            "[Discharging] cycle raw count: %s, capacity raw count: %s"
            % (len(self.discharge_cyc_raw), len(self.discharge_cap_raw))
        )

    # =================================================================
    # === REFACTORED METHODS START HERE ===
    # =================================================================

    def prepare_data(self, train_names, test_names, validation_names=None):
        """
        Prepares data for all specified splits (train, test, and optional validation).
        """
        logger.debug(
            f"Start preparing data for training: {train_names}, testing: {test_names}..."
        )
        if validation_names:
            logger.debug(f"Including validation data: {validation_names}...")

        # --- Use dictionaries to store data for each split ---
        self.charge_cyc = {}
        self.charge_cap = {}
        self.discharge_cyc = {}
        self.discharge_cap = {}

        # Define the splits to process
        splits_to_process = {"train": train_names, "test": test_names}
        if validation_names:
            splits_to_process["val"] = validation_names

        # --- Loop through splits instead of repeating code ---
        for split_name, names in splits_to_process.items():
            if not names:  # Skip if a name list is empty or None
                logger.debug(f"Skipping {split_name} split as no names were provided.")
                continue

            logger.debug(f"--- Processing {split_name} split ---")

            # Get cycle and capacity data
            self.charge_cyc[split_name], self.charge_cap[split_name] = (
                self.__get_cyc_and_cap(names, self.charge_cyc_raw, self.charge_cap_raw)
            )
            self.discharge_cyc[split_name], self.discharge_cap[split_name] = (
                self.__get_cyc_and_cap(
                    names, self.discharge_cyc_raw, self.discharge_cap_raw
                )
            )
            logger.debug(f"Finish getting {split_name} charge and discharge data.")

            # Clean charge data
            (
                self.charge_cyc[split_name],
                self.charge_cap[split_name],
            ) = self.__clean_cyc_and_cap_without_mapping(
                self.charge_cyc[split_name],
                self.charge_cap[split_name],
                self.discharge_cap[split_name],
            )
            logger.debug(f"Finish cleaning {split_name} charge data.")

            # Clean discharge data
            (
                self.discharge_cyc[split_name],
                self.discharge_cap[split_name],
            ) = self.__clean_cyc_and_cap_without_mapping(
                self.discharge_cyc[split_name],
                self.discharge_cap[split_name],
                self.charge_cap[split_name],
            )
            logger.debug(f"Finish cleaning {split_name} discharge data.")

            # Add discharge parameters
            self.discharge_cyc[split_name] = self.__add_discharge_soc_pars(
                self.discharge_cyc[split_name], self.charge_cap[split_name]
            )
            self.discharge_cap[split_name] = self.__add_discharge_soh_pars(
                self.discharge_cap[split_name], self.charge_cap[split_name]
            )
            logger.debug(f"Finish adding {split_name} discharge SOC/SOH parameters.")

            # Log final shapes
            logger.info(
                f"Prepared {split_name} charge cycle data: {self.charge_cyc[split_name].shape}, "
                f"capacity data: {self.charge_cap[split_name].shape}"
            )
            logger.info(
                f"Prepared {split_name} discharge cycle data: {self.discharge_cyc[split_name].shape}, "
                f"capacity data: {self.discharge_cap[split_name].shape}"
            )

        logger.debug("Finish preparing all data.")

    # =================================================================
    # === ORIGINAL HELPER METHODS (UNCHANGED) ===
    # =================================================================

    def __get_cyc_and_cap(self, names, cyc_raw, cap_raw):
        cyc_data = []
        cap_data = []

        gp_cyc_raw = self.__group_cyc_by_name(cyc_raw, names)

        gp_cap_raw = cap_raw.groupby("test_name")

        for test_name in names:
            last_cap_group_index = 0
            cap_group = gp_cap_raw.get_group(test_name).reset_index(drop=True)

            for cycle in gp_cyc_raw[test_name]:
                cycle = cycle.reset_index(drop=True)
                cycle_count = cycle.iloc[-1]["cycle_count"]

                target_cap_row_indices = np.array(
                    cap_group.index[(cap_group["cycle_count"] == cycle_count)]
                )

                # Handle cases where cycle might not have a corresponding capacity record
                valid_indices = target_cap_row_indices[
                    target_cap_row_indices >= last_cap_group_index
                ]

                if len(valid_indices) == 0:
                    # Log a warning or skip this cycle
                    logger.warning(
                        f"No matching capacity record found for test {test_name}, cycle {cycle_count}. Skipping."
                    )
                    continue

                target_cap_row_index = valid_indices[0]
                target_cap_row = cap_group.iloc[target_cap_row_index]

                last_cap_group_index = target_cap_row_index

                cyc_data.append(cycle.values)
                cap_data.append(target_cap_row.values)

        cyc_data = np.array(cyc_data, dtype=object)
        cap_data = np.array(cap_data, dtype=object)

        # Make sure cyc_data has two dim
        # cyc_data = cyc_data.reshape(-1,1)
        return (cyc_data, cap_data)

    def __group_cyc_by_name_and_cyc_count(self, cyc_raw):
        return cyc_raw.groupby(
            [
                "test_name",
                (cyc_raw["cycle_count"] != cyc_raw["cycle_count"].shift()).cumsum(),
            ]
        )

    def __group_cyc_by_name(self, cyc_raw, test_names):
        grouped_cycle = self.__group_cyc_by_name_and_cyc_count(cyc_raw)
        grouped_name_cycle = {}
        for key, group in grouped_cycle:
            test_name = key[0]
            if test_name not in grouped_name_cycle:
                grouped_name_cycle[test_name] = []
            grouped_name_cycle[test_name].append(group)
        return grouped_name_cycle

    def __clean_cyc_and_cap_without_mapping(self, target_cyc, target_cap, mapping_cap):
        """Clean all charge/discharge cycle which does not have corresponding mapping discharge/charge cycle"""
        clean_indices = []
        dirty_row = 0

        # Handle empty arrays
        if len(target_cyc) == 0 or len(mapping_cap) == 0:
            return (target_cyc, target_cap)

        for i in range(len(target_cyc)):
            if (
                i >= len(mapping_cap)
                or target_cap[i][CapacityCols.CYCLE_COUNT]
                != mapping_cap[i - dirty_row][CapacityCols.CYCLE_COUNT]
            ):
                dirty_row += 1
            else:
                clean_indices.append(i)

        return (target_cyc[clean_indices], target_cap[clean_indices])

    def __add_discharge_soc_pars(self, discharge_cyc, charge_cap):
        for i in range(len(discharge_cyc)):
            # SOC: (last charge cycle capacity - discharging capacity) / last charge cycle capacity
            discharge_cyc[i] = np.c_[
                discharge_cyc[i], np.zeros(discharge_cyc[i].shape[0])
            ]
            discharge_cyc[i][:, -1] = (
                charge_cap[i][CapacityCols.CHARGING_CAPACITY]
                - discharge_cyc[i][:, CycleCols.DISCHARGING_CAPACITY]
            ) / charge_cap[i][CapacityCols.CHARGING_CAPACITY]

            # Time remaining to cycle end: (Time of last row in cycle - current time)
            discharge_cyc[i] = np.c_[
                discharge_cyc[i], np.zeros(discharge_cyc[i].shape[0])
            ]
            discharge_cyc[i][:, -1] = (
                discharge_cyc[i][-1:, CapacityCols.TIME]
                - discharge_cyc[i][:, CycleCols.TIME]
            )

        return discharge_cyc

    def __add_discharge_soh_pars(self, discharge_cap, charge_cap):
        if discharge_cap.shape[0] == 0:
            return discharge_cap  # Return empty if no data

        discharge_cap = np.c_[discharge_cap, np.zeros((discharge_cap.shape[0], 5))]

        for cap in discharge_cap:
            # Time remaining to cell end: (Time of last row in the cell - current time)
            cap[CapacityCols.REMAINING_TIME_TO_CELL_END] = (
                discharge_cap[
                    discharge_cap[:, CapacityCols.TEST_NAME]
                    == cap[CapacityCols.TEST_NAME]
                ][-1][CapacityCols.TIME]
                - cap[CapacityCols.TIME]
            )

            # Maximum capacity in corresponding charging cycles
            same_test_charge_cap = charge_cap[
                charge_cap[:, CapacityCols.TEST_NAME] == cap[CapacityCols.TEST_NAME]
            ]
            if same_test_charge_cap.shape[0] > 0:
                cap[CapacityCols.MAXIMUM_CAPACITY] = np.max(
                    same_test_charge_cap[:, CapacityCols.CHARGING_CAPACITY]
                )
            else:
                cap[CapacityCols.MAXIMUM_CAPACITY] = (
                    np.nan
                )  # Avoid error on empty slice

            # Nominal cell capacity
            cell_cap = cap[CapacityCols.MAXIMUM_CAPACITY]
            # Test name convention: 000-XW-Y.Y-AABB-T (7~10 chars are cell capacity)
            cell_cap_text = cap[CapacityCols.TEST_NAME][7:10]
            try:
                cell_cap = float(cell_cap_text)
            except Exception:
                pass
            cap[CapacityCols.NOMINAL_CAPACITY] = cell_cap

        # SOH: (Last charging cycle capacity / nominal cell capacity)
        discharge_cap[:, CapacityCols.SOH] = (
            charge_cap[:, CapacityCols.CHARGING_CAPACITY]
            / discharge_cap[:, CapacityCols.MAXIMUM_CAPACITY]
        )

        # Corresponding charging cycle charging capacity
        discharge_cap[:, CapacityCols.CORRESPONDING_CHARGING_CAPACITY] = charge_cap[
            :, CapacityCols.CHARGING_CAPACITY
        ]

        return discharge_cap

    def get_charge_data(self):
        """
        Returns dictionaries for cycle and capacity data, keyed by split name.
        Example: cyc_data['train'], cap_data['train']
        """
        return self.charge_cyc, self.charge_cap

    def get_discharge_data(self):
        """
        Returns dictionaries for cycle and capacity data, keyed by split name.
        Example: cyc_data['train'], cap_data['train']
        """
        return self.discharge_cyc, self.discharge_cap

    def get_all_test_names(self):
        return self.cycle_raw["test_name"].unique()
