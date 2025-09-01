import warnings
import logging
from lmetk.data.datasources import SingleSourceLoader
from lmetk.data.datasources.kubric_movie.movi_dataset import MoviDataset
from collections import defaultdict
from lmetk.data.datasources.base.interfaces import AbstractDataSourceLoader
from lmetk.data.data_objects.data import SplitDatasetContainer
from typing import Any, Dict, override
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


from pympler import asizeof

def sizeof_gb(obj) -> float:
    """Return the memory usage of a Python object in GB (deep, recursive)."""
    return asizeof.asizeof(obj) / (1024 ** 3)

def convert_outer_list_to_inner(data_list):
    if not data_list:
        return {}

    # Ensure all dictionaries have the same keys
    keys = data_list[0].keys()
    for d in data_list:
        if d.keys() != keys:
            raise ValueError("Not all dicts have the same keys!")

    # Concatenate values for each key
    stacked = {}
    for key in keys:
        stacked[key] = [d[key] for d in data_list]

    # Check that all keys in the stacked have the same length
    lengths = [len(stacked[key]) for key in stacked]
    if len(set(lengths)) != 1:
        raise ValueError("All keys in the stacked dict must have the same length.")

    return stacked

class MOViLoader(AbstractDataSourceLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_path = kwargs['data_path']
        self.normalization_file = kwargs['normalization_file']
        
        self._is_isplitted = False
        self._is_loaded = False
        self.data_dict = None
        self.meta_data = None

        # Instantiate multisource_data_splitter if is not none
        assert (
            kwargs.get("multisource_data_splitter") == None
        ), "For RigidBodyDynanics, multisource_data_splitter is not supported. RigidBodyDynanics are given with splits."
        self.multisource_data_splitter = "For RigidBodyDynanics, multisource_data_splitter is not supported. RigidBodyDynanics are given with splits."
    
    @override
    def load_data(self):
        self.data_dict = self._load_data()
        self._is_loaded = True
        self._is_isplitted = True
    
    @override
    def get_data(self) -> dict | list[dict]:
        """Return data sources.

        Returns:
            dict | list[dict] : either dict in case we use multisource_data_splitter or list of dicts in case we use data_splitter

        """
        assert self._is_loaded is True, "Data must be loaded before getting data."
        assert self._is_isplitted is True, "Data must be splitted before getting data."
        return SplitDatasetContainer(**self.data_dict)

    def _load_data(self) -> dict:
        # Read raw CSV
        data = self.read_data()

        size_gb = sizeof_gb(data) / (1024**3)
        logger.info(f"{size_gb:.6f} GB")
        # Convert to desired format
        train = convert_outer_list_to_inner(data["train"])
        val = convert_outer_list_to_inner(data["val"])
        test = convert_outer_list_to_inner(data["test"])
        
        out_dict = {}
        for key in train.keys():
            out_dict[key] = {
                "train": train[key],
                "val": val[key],
                "test": test[key],
            }

        logger.info(f"Number of train units: {len(out_dict['features']['train'])}")
        logger.info(f"Number of val units: {len(out_dict['features']['val'])}")
        logger.info(f"Number of test units: {len(out_dict['features']['test'])}")

    def read_data(
        self,
    ):
        data = defaultdict(list)
        for split in ["train", "val", "test"]:
            loader = MoviDataset(
                dataset_dir=self.data_path, 
                normalization_file=self.normalization_file, 
                nox4=False,
                split=split)
            data[split].extend([loader.__getitem__(i) for i in range(loader.__len__())])

        

        return data

    def get_meta_data(self) -> Dict[str, Any]:
        """
        Return metadata about the data source.
        This can include information like data source name, version, etc.
        """
        assert self._is_loaded, "Data must be loaded before getting metadata."
        return self.meta_data

    def get_data_name(self) -> list[str]:
        """Return source data names."""
        return self.data_dict["metadata"]

    def get_multisource_data_splitter(self):
        """Return multisource data splitter."""
        return self.multisource_data_splitter

    @override
    def split_data(self):
        logger.warning(
            "RigidBodyDynamics does not support data splitting. The project relies on predefined splits."
        )