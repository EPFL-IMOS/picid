import copy
import logging
import os
import warnings
from typing import Any, Dict, override

import awkward as ak
import h5py
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Internal library imports
from lmetk.data.data_objects.data import SplitDatasetContainer
from lmetk.data.datasources.base.interfaces import AbstractDataSourceLoader
from .utils import convert_outer_list_to_inner

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class AirbusHelicopterLoader(AbstractDataSourceLoader):
    """
    Loader for the Airbus Helicopter Accelerometer Dataset.

    Automatically downloads the dataset if not present in `data_dir`.

    Loads data in a ragged representation:
    - Features: (N, 60, 1024, 1) corresponding to (Seq, Seconds, Hz, Channels)
    - Targets: (N, 60, 1024, 1) broadcasted label

    Expected file structure in `data_dir`:
      - training_healthy.h5 (1677 samples)
      - dataset_anomalies.h5 (594 samples)
      - ground_truth.csv
    """

    # Direct download links from ETH Zurich Research Collection
    _URLS = {
        "training_healthy.h5": "https://www.research-collection.ethz.ch/bitstreams/a701c2e4-a3ca-41e4-99b9-2fa657fd72bf/download",
        "dataset_anomalies.h5": "https://www.research-collection.ethz.ch/bitstreams/34ecff00-0cf8-4891-ad5f-b347396ac14b/download",
        "ground_truth.csv": "https://www.research-collection.ethz.ch/bitstreams/2c21d0db-da85-4e05-be87-4ad44a3647b6/download",
    }

    def __init__(
        self,
        data_dir: str,
        data_name: str,
        task_mode: str,
        multisource_data_splitter: Any = None,
        download: bool = True,  # Added flag
        **kwargs,
    ):
        """
        Args:
            data_dir (str): Path to the directory where data will be stored/loaded.
            data_name (str): Name of the dataset.
            task_mode (str): Task type (e.g. 'detection').
            download (bool): If True, downloads files if missing. Default True.
        """
        super().__init__(
            data_name=data_name,
            task_mode=task_mode,
            multisource_data_splitter=multisource_data_splitter,
            **kwargs,
        )

        self.data_path = data_dir
        self.should_download = download

        self._is_isplitted = False
        self._is_loaded = False
        self.data_dict = None
        self.meta_data = {}

        if multisource_data_splitter is not None:
            logger.warning(
                f"{self.data_name} does not support multisource splitting. "
                "The dataset relies on predefined file splits."
            )
        self.multisource_data_splitter = None

    @override
    def load_data(self):
        """Loads and structures the data, downloading if necessary."""
        # Ensure directory exists
        os.makedirs(self.data_path, exist_ok=True)

        # Check and download files
        if self.should_download:
            self._download_missing_files()

        self.data_dict = self._load_data()
        self._is_loaded = True
        self._is_isplitted = True

    @override
    def get_data(self) -> SplitDatasetContainer:
        """Returns the formatted data container."""
        assert self._is_loaded is True, "Data must be loaded before getting data."
        assert self._is_isplitted is True, "Data must be splitted before getting data."
        return SplitDatasetContainer(**self.data_dict)

    def _download_missing_files(self):
        """Checks for files and downloads them if missing."""
        for filename, url in self._URLS.items():
            file_path = os.path.join(self.data_path, filename)

            if not os.path.exists(file_path):
                logger.info(f"Downloading {filename} to {self.data_path}...")
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()

                    # Get total size for progress bar
                    total_size = int(response.headers.get("content-length", 0))
                    block_size = 1024 * 1024  # 1MB chunks

                    with (
                        open(file_path, "wb") as f,
                        tqdm(
                            desc=filename,
                            total=total_size,
                            unit="iB",
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as bar,
                    ):
                        for chunk in response.iter_content(chunk_size=block_size):
                            size = f.write(chunk)
                            bar.update(size)

                    logger.info(f"Successfully downloaded {filename}")
                except Exception as e:
                    logger.error(f"Failed to download {filename}: {e}")
                    # Cleanup partial download
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    raise e
            else:
                logger.info(f"File {filename} already exists. Skipping download.")

    def _load_data(self) -> dict:
        """
        Orchestrates the reading and formatting of data into the
        standard {train, val, test} structure.
        """
        # 1. Read raw data from disk
        raw_data_splits = self.read_data()

        # 2. Convert list of dicts -> dict of lists (Columnar format)
        train = convert_outer_list_to_inner(raw_data_splits["train"])
        test = convert_outer_list_to_inner(raw_data_splits["test"])

        # 3. Create Validation Split (Deep copy of test)
        val = convert_outer_list_to_inner(copy.deepcopy(raw_data_splits["test"]))

        # --- SIZE ASSERTIONS ---
        n_train = len(train.get("features", []))
        n_test = len(test.get("features", []))
        n_val = len(val.get("features", []))

        assert n_train == 1677, f"Expected 1677 training samples, found {n_train}"
        assert n_test == 594, f"Expected 594 test samples, found {n_test}"
        assert n_val == 594, f"Expected 594 validation samples, found {n_val}"

        # 4. Structure into the final output format
        out_dict = {}
        keys = train.keys() if train else test.keys()

        for key in keys:
            out_dict[key] = {
                "train": train.get(key, []),
                "val": val.get(key, []),
                "test": test.get(key, []),
            }

        # Logging stats
        logger.info(
            f"Loaded Airbus Data: {n_train} train, {n_test} test, {n_val} validation sequences."
        )

        # 5. Extract and store metadata
        unit_ids = out_dict.get("unit_id", {})
        metadata_vals = out_dict.get("metadata", {})

        self.meta_data.update(
            {
                "unit_ids": {
                    "train": unit_ids.get("train", []),
                    "val": unit_ids.get("val", []),
                    "test": unit_ids.get("test", []),
                },
                "unit_names": {
                    "train": [m["unit_name"] for m in metadata_vals.get("train", [])],
                    "val": [m["unit_name"] for m in metadata_vals.get("val", [])],
                    "test": [m["unit_name"] for m in metadata_vals.get("test", [])],
                },
                "dims_explanation": (
                    "Ragged representation: Features shape (60, 1024, 1) -> 60 seconds, 1024 Hz, 1 Feature. "
                    "Targets are broadcasted to match features shape."
                ),
            }
        )

        return out_dict

    def read_data(self) -> Dict[str, list]:
        """Reads files, inspects them, and converts to processed list of dicts."""
        train_file = os.path.join(self.data_path, "training_healthy.h5")
        test_file = os.path.join(self.data_path, "dataset_anomalies.h5")
        gt_file = os.path.join(self.data_path, "ground_truth.csv")

        # 1. Load Ground Truth Map
        if os.path.exists(gt_file):
            df_gt = pd.read_csv(gt_file)
            if "seqID" in df_gt.columns:
                gt_map = dict(zip(df_gt["seqID"], df_gt["anomaly"]))
            else:
                gt_map = dict(zip(df_gt.index, df_gt["anomaly"]))
        else:
            logger.warning(
                f"Ground truth file not found at {gt_file}. Assuming unlabeled."
            )
            gt_map = {}

        split_data = {"train": [], "test": []}

        # 2. Process Train Data
        if os.path.exists(train_file):
            logger.info(f"Reading {train_file}...")
            df_train = pd.read_hdf(train_file, key="dftrain")
            split_data["train"] = self._process_dataframe(
                df_train, split_name="train", gt_map=None
            )
        else:
            raise FileNotFoundError(f"Training file not found: {train_file}")

        # 3. Process Test Data
        if os.path.exists(test_file):
            logger.info(f"Reading {test_file}...")
            df_test = pd.read_hdf(test_file, key="dfvalid")
            split_data["test"] = self._process_dataframe(
                df_test, split_name="test", gt_map=gt_map
            )
        else:
            raise FileNotFoundError(f"Test/Anomaly file not found: {test_file}")

        return split_data

    def _process_dataframe(
        self, df: pd.DataFrame, split_name: str, gt_map: Dict = None
    ) -> list:
        """Converts the loaded pandas DataFrame into the list-of-dicts format."""
        processed_list = []

        SAMPLING_FREQ = 1024  # Hz
        DURATION_SEC = 60  # seconds
        EXPECTED_LENGTH = SAMPLING_FREQ * DURATION_SEC  # 61440

        iterator = tqdm(df.values, desc=f"Processing {split_name}", unit="seq")

        for idx, row_values in enumerate(iterator):
            # --- ASSERTIONS ---
            assert len(row_values) == EXPECTED_LENGTH, (
                f"Sequence {idx} in {split_name} has invalid length {len(row_values)}. "
                f"Expected {EXPECTED_LENGTH}."
            )

            # --- 1. FEATURE PROCESSING ---
            features_np = row_values.reshape(DURATION_SEC, SAMPLING_FREQ, 1)

            assert features_np.shape == (DURATION_SEC, SAMPLING_FREQ, 1), (
                f"Reshape failed for sequence {idx}. " f"Got {features_np.shape}"
            )

            features_ak = ak.from_regular(ak.from_numpy(features_np), axis=1)

            # --- 2. TARGET PROCESSING ---
            if split_name == "train":
                label = 0.0  # Healthy
                unit_name = f"train_seq_{idx}"
            else:
                label = gt_map.get(idx, float("nan"))
                unit_name = f"test_seq_{idx}"

            unit_id = idx

            target_np = np.full(
                (DURATION_SEC, SAMPLING_FREQ, 1), label, dtype=np.float32
            )

            assert target_np.shape == (
                DURATION_SEC,
                SAMPLING_FREQ,
                1,
            ), "Target shape mismatch"

            target_ak = ak.from_regular(ak.from_numpy(target_np), axis=1)

            # --- 3. ENTRY CONSTRUCTION ---
            entry = {
                "features": features_ak,
                "target": target_ak,
                "unit_id": np.array([unit_id]),
                "metadata": {
                    "unit_name": unit_name,
                    "n_cycles": len(row_values),
                    "split": split_name,
                    "sampling_freq": SAMPLING_FREQ,
                    "duration": DURATION_SEC,
                    "dims_info": "60s x 1024Hz x 1feat",
                },
            }

            if self.debug_subsample_rate is not None:
                entry["features"] = entry["features"][:: self.debug_subsample_rate]
                entry["target"] = entry["target"][:: self.debug_subsample_rate]

            processed_list.append(entry)

        return processed_list

    def get_meta_data(self) -> Dict[str, Any]:
        assert self._is_loaded, "Data must be loaded before getting metadata."
        return self.meta_data

    def get_data_name(self) -> str:
        return self.data_name

    @override
    def split_data(self):
        logger.warning(
            f"{self.data_name} relies on predefined splits. split_data() has no effect."
        )
