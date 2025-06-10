
# import numpy as np
# import pandas as pd
# from typing import Dict, List, Optional, Tuple

# from lmetk.data.datasources.base import DataSourceLoader
# from .base import TimeSeriesSplitter



# class BySourceSplitter:
#     """
#     Splitter that splits the dataset by source.
#     """

#     def __init__(self, 
#                 sources_train: List[str] = None,
#                 sources_val: List[str] = None,
#                 sources_test: List[str] = None):
        
        
#         self.sources_train = sources_train if sources_train is not None else []
#         self.sources_val = sources_val if sources_val is not None else []
#         self.sources_test = sources_test if sources_test is not None else []

#         # make sure sources are disjoint sets
#         all_sources = set(self.sources_train) | set(self.sources_val) | set(self.sources_test)
#         if len(all_sources) != (len(self.sources_train) + len(self.sources_val) + len(self.sources_test)):
#             raise ValueError("Sources must be disjoint sets. Some sources appear in multiple splits.")

#     def split_data(self, sources: Dict[str, DataSourceLoader]) -> Dict[str, Dict[str, DataSourceLoader]]:

#         """
#         Split the data by source.

#         Parameters
#         ----------
#         sources : Dict[str, DataSourceLoader]
#             Dictionary containing DataSourceLoader instances for each source.

#         Returns
#         -------
#         Dict[str, Dict[np.ndarray]]
#             Dictionary containing split data arrays for each source.
#         """
#         split_data = {
#             "train": {},
#             "val": {},
#             "test": {}
#         }
        
#         # TODO: how to get masks here? 
#         split_data_mask = {
#             "train": {},
#             "val": {},
#             "test": {}
#         }

#         for split, sources_list in zip(
#             ["train", "val", "test"],
#             [self.sources_train, self.sources_val, self.sources_test]
#         ):
#             for s in sources_list:
#                 if s in sources:
#                     split_data[split][s] = sources[s]
                    
#                 else:
#                     raise ValueError(f"Source {s} not found in provided sources.")
            

#         # Print a tree view of the splits
#         print("Split tree view:")
#         for split in ["train", "val", "test"]:
#             print(f"|- {split}")
#             for source in split_data[split]:
#                 print(f"   |- {source}")

#         return split_data