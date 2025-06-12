
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from lmetk.data.datasources.base import DataSourceLoader




class BySourceSplitter:
    """
    Splitter that splits the dataset by source.
    """

    def __init__(self, 
                sources_train: List[str] = None,
                sources_val: List[str] = None,
                sources_test: List[str] = None):
        
        
        self.sources_train = sources_train if sources_train is not None else []
        self.sources_val = sources_val if sources_val is not None else []
        self.sources_test = sources_test if sources_test is not None else []

        # make sure sources are disjoint sets
        all_sources = set(self.sources_train) | set(self.sources_val) | set(self.sources_test)
        if len(all_sources) != (len(self.sources_train) + len(self.sources_val) + len(self.sources_test)):
            raise ValueError("Sources must be disjoint sets. Some sources appear in multiple splits.")

    def split_data(self, data_list: List[Dict[str, Any]], source_name_lst: List[str]) -> List[Dict[str, Any]]:

        """
        Split the data by source.

        Parameters
        ----------
        data_list : Dict[str, Any]
            List containing data dicts instances for each source.

        Returns
        -------
        data_list : Dict[str, Any]
            List containing splitted data dicts instance.
        """

        if len(data_list) != len(source_name_lst):
            raise ValueError(f"Length of data_list={len(data_list)} and source_name_lst={len(source_name_lst)} must be equal.")

        sources = dict(zip(source_name_lst, data_list))

        
        split_data = {
            key: {'train': None, 'val': None, 'test': None}
            for key in data_list[0].keys()
        }

        for split, split_sources_list in zip(
            ["train", "val", "test"],
            [self.sources_train, self.sources_val, self.sources_test]
        ):
            # Collect data dicts from selected sources
            selected_data_dicts = [sources[key] for key in split_sources_list]

            # Stack them
            stacked = stack_dicts_with_concat(selected_data_dicts)

            # Assign each stacked key to its correct place in split_data
            for key in stacked:
                split_data[key][split] = stacked[key]


        print("Split tree view:")
        for split, split_sources_list in zip(
            ["train", "val", "test"],
            [self.sources_train, self.sources_val, self.sources_test]
        ):
            print(f"|- {split}")
            for source in split_sources_list:
                print(f"   |- {source}")

        return split_data
    
    def __call__(self, data_list: List[Dict[str, Any]], source_name_lst: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
        sources_by_split = {
            "train": self.sources_train,
            "val": self.sources_val,
            "test": self.sources_test
        }
        
        return [self.split_data(data_list, source_name_lst)], [self.get_split_tree_view(sources_by_split)]

    def get_split_tree_view(self, sources_by_split: dict[str, list[str]]) -> str:
        lines = ["Split tree view:"]
        for split in ["train", "val", "test"]:
            lines.append(f"|- {split}")
            for source in sources_by_split.get(split, []):
                lines.append(f"   |- {source}")
        return "\n".join(lines)

def stack_dicts_with_concat(data_list):
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
        try:
            stacked[key] = np.concatenate([d[key] for d in data_list], axis=0)
        except Exception as e:
            raise ValueError(f"Failed to concatenate key '{key}': {e}")
    return stacked  