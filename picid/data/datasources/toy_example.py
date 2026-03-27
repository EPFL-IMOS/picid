import logging
import warnings
from typing import Any, Dict, List, override

import awkward as ak
import numpy as np
from tqdm import tqdm

# --- LIBRARY IMPORTS ---
# These are the base classes you must inherit from or use
from picid.data.data_objects import SplitDatasetContainer
from picid.data.datasources.base.interfaces import AbstractDataSourceLoader


# In your actual code, you import this from utils.py
# We define it here locally just to make this example self-contained
def convert_outer_list_to_inner(data_list: list[dict]) -> dict:
    if not data_list:
        return {}
    keys = data_list[0].keys()
    return {key: [d[key] for d in data_list] for key in keys}


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class ToyRaggedLoader(AbstractDataSourceLoader):
    """
    A TOY EXAMPLE implementation of a DataSourceLoader.

    OBJECTIVE:
    This class demonstrates how to add a dataset where sequences have DIFFERENT lengths
    (Ragged Data), which is a core feature of this library.

    SCENARIO:
    - We simulate a 'Vibration' dataset.
    - 'Train' contains only Healthy signals (Target = 0).
    - 'Test' contains Anomalous signals (Target = 1).
    - Crucially: Each recording has a different duration!

    DATA STRUCTURE EXPLAINED:
    The library expects data in a specific "Ragged" format using Awkward Arrays:

    1. FEATURES SHAPE: (Time, Frequency/Samples, Channels)
       - Example: (60, 1024, 1) means 60 seconds, 1024 Hz sampling, 1 sensor.
       - If your data is just a flat 1D signal, you must reshape it to (Time, Freq, 1).

    2. TARGETS SHAPE: (Time, 1, 1)
       - We optimize memory by storing 1 label per time-step (e.g., per second)
       - Instead of repeating the label 1024 times per second, we store it once.
       - This is "broadcasted" automatically by downstream logic if needed.
    """

    def __init__(
        self,
        data_dir: str,
        data_name: str = "toy_ragged_dataset",
        task_mode: str = "anomaly_detection",
        multisource_data_splitter: Any = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        data_dir
            Where data is stored (not used in this toy example as we generate data).
        data_name
            Unique identifier for the dataset.
        task_mode
            'anomaly_detection', 'diagnosis', or 'rul'.
        """
        super().__init__(
            data_name=data_name,
            task_mode=task_mode,
            multisource_data_splitter=multisource_data_splitter,
            **kwargs,
        )

        # Standard boilerplate flags
        self.data_path = data_dir
        self._is_splitted = False
        self._is_loaded = False
        self.data_dict = None
        self.meta_data = {}
        self.multisource_data_splitter = None  # We handle splits manually here

    @override
    def load_data(self):
        """
        Public method called by the pipeline to trigger loading.
        1. Reads/Generates raw data.
        2. Formats it into the library's internal structure.
        """
        print(f"[{self.data_name}] Loading Started...")
        self.data_dict = self._load_data()
        self._is_loaded = True
        self._is_splitted = True
        print(f"[{self.data_name}] Loading Complete.")

    @override
    def get_data(self) -> SplitDatasetContainer:
        """Wraps the loaded dictionary into the standardized Container object."""
        assert self._is_loaded, "Data must be loaded first!"
        return SplitDatasetContainer(**self.data_dict)

    def _load_data(self) -> dict:
        """
        INTERNAL LOGIC:
        This is where the magic happens. We convert raw numpy arrays into
        the library's List-of-Dicts structure.
        """
        # 1. READ RAW DATA
        # In a real scenario, you would read .csv or .h5 files here.
        # Here, we generate synthetic ragged data.
        raw_splits = self.generate_synthetic_data()

        # 2. PROCESS SPLITS
        # We process each split (Train/Test) to create Awkward Arrays.
        # Note: We use the helper 'convert_outer_list_to_inner' to flip
        # [ {feat, targ}, {feat, targ} ]  ->  { feat: [], targ: [] }

        train = convert_outer_list_to_inner(
            self._process_raw_list(raw_splits["train"], split_name="train")
        )

        test = convert_outer_list_to_inner(
            self._process_raw_list(raw_splits["test"], split_name="test")
        )

        # Validation is often just a copy or subset of test for anomaly detection
        val = convert_outer_list_to_inner(
            self._process_raw_list(
                raw_splits["test"], split_name="val"
            )  # Just reusing test data for demo
        )

        # 3. BUILD OUTPUT DICTIONARY
        out_dict = {}
        keys = train.keys()  # ['features', 'target', 'unit_id', 'metadata']

        for key in keys:
            out_dict[key] = {
                "train": train.get(key, []),
                "val": val.get(key, []),
                "test": test.get(key, []),
            }

        # 4. METADATA (Crucial for downstream transforms)
        # You must store unit IDs and names so the pipeline can track them.
        self.meta_data.update(
            {
                "unit_ids": {
                    "train": out_dict["unit_id"]["train"],
                    "val": out_dict["unit_id"]["val"],
                    "test": out_dict["unit_id"]["test"],
                },
                "unit_names": {
                    "train": [m["unit_name"] for m in out_dict["metadata"]["train"]],
                    "val": [m["unit_name"] for m in out_dict["metadata"]["val"]],
                    "test": [m["unit_name"] for m in out_dict["metadata"]["test"]],
                },
                "dims_explanation": "Features: (Time, 100Hz, 1). Targets: (Time, 1, 1).",
            }
        )

        return out_dict

    def _process_raw_list(
        self, raw_data_list: List[np.ndarray], split_name: str
    ) -> List[Dict]:
        """
        Converts a list of raw numpy arrays into the library's dictionary format.
        """
        processed_list = []

        # HARDCODED CONSTANTS (Specific to your dataset physics)
        SAMPLING_FREQ = 100  # Hz (Low freq for toy example)

        desc = f"Processing {split_name}"
        for idx, raw_signal in enumerate(tqdm(raw_data_list, desc=desc)):
            # --- 1. HANDLING RAGGEDNESS ---
            # Raw signal shape: (N_samples,) -> e.g., (6000,) or (4500,)
            # We want to reshape this into (Time, Freq, Channels)

            total_points = len(raw_signal)

            # Calculate how many full seconds we have
            n_seconds = total_points // SAMPLING_FREQ

            # Truncate to full seconds (optional, depends on your needs)
            truncated_len = n_seconds * SAMPLING_FREQ
            raw_signal = raw_signal[:truncated_len]

            # RESHAPE: (Time, Freq, 1)
            # This makes "Time" the ragged dimension (variable length)
            # "Freq" (100) and "Channels" (1) are fixed dimensions.
            features_np = raw_signal.reshape(n_seconds, SAMPLING_FREQ, 1)

            # Convert to Awkward Array
            # axis=1 preserves the first dimension (n_seconds) as the outer array length
            features_ak = ak.from_regular(ak.from_numpy(features_np), axis=1)

            # --- 2. CREATE TARGETS ---
            # Determine label based on split (Simple logic for toy example)
            # Train = 0 (Healthy), Test = 1 (Anomaly)
            label = 0.0 if split_name == "train" else 1.0

            # TARGET SHAPE OPTIMIZATION: (Time, 1, 1)
            # We create 1 label per second.
            target_np = np.full((n_seconds, 1, 1), label, dtype=np.float32)
            target_ak = ak.from_regular(ak.from_numpy(target_np), axis=1)

            # --- 3. CREATE METADATA ENTRY ---
            # Every sample is a dictionary
            entry = {
                "features": features_ak,
                "target": target_ak,
                "unit_id": np.array([idx]),  # Unique ID
                "metadata": {
                    "unit_name": f"{split_name}_{idx}",
                    "raw_length": total_points,
                    "duration_sec": n_seconds,
                    "split": split_name,
                },
            }

            processed_list.append(entry)

        return processed_list

    def generate_synthetic_data(self):
        """
        Generates dummy data to make this example runnable.

        Returns
        -------
        dict
            Dict with 'train' and 'test' lists of numpy arrays.
        """
        print("Generating Synthetic Ragged Data...")
        rng = np.random.default_rng(42)

        # Train: 5 sequences, varying lengths between 300 and 500 samples
        train_data = [rng.random(rng.integers(300, 500)) for _ in range(5)]

        # Test: 3 sequences, varying lengths between 100 and 600 samples
        test_data = [rng.random(rng.integers(100, 600)) for _ in range(3)]

        return {"train": train_data, "test": test_data}

    # Required Abstract Methods
    @override
    def get_meta_data(self) -> Dict[str, Any]:
        return self.meta_data

    @override
    def get_data_name(self) -> str:
        return self.data_name

    @override
    def split_data(self):
        # Only needed if you use the 'multisource_data_splitter' logic.
        logger.warning("Split data not implemented for predefined splits.")


# =============================================================================
# MAIN EXECUTION (DEBUGGING / VERIFICATION)
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Running ToyRaggedLoader Example")
    print("=" * 60)

    # 1. Instantiate
    # In Hydra, this would be: dataset_source = hydra.utils.instantiate(cfg.datasource)
    loader = ToyRaggedLoader(
        data_dir="./dummy_path",
        task_mode="anomaly_detection",  # Ignored by this toy class
    )

    # 2. Load
    loader.load_data()

    # 3. Get Container
    container = loader.get_data()

    # 4. Inspect Structure
    print("\n--- INSPECTION ---")

    # CORRECT ACCESS: Use container.features['train'], NOT container.train['features']
    train_feats = container.features["train"]
    train_targs = container.target["train"]

    print(f"Number of Train Units: {len(train_feats)}")

    # Demonstrate Raggedness
    print("\n[Raggedness Verification]")
    for i in range(len(train_feats)):
        # Convert Awkward -> Numpy for inspection
        # Shape: (Time, 100, 1)
        shape = train_feats[i].to_numpy().shape
        label_shape = train_targs[i].to_numpy().shape

        print(f"Unit {i}: Feature Shape {shape} | Target Shape {label_shape}")

    print("\nNotice how the first dimension (Time) varies per unit!")
    print("This is the power of the Ragged representation.")
    print("=" * 60)
