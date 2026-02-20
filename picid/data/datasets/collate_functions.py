import torch
from collections import defaultdict
from typing import List, Dict, Any


def collate_key_value_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function that merges a list of dictionaries into a single dictionary.
    Each key in the resulting dictionary maps to a list of values collected from
    the corresponding keys in the input dictionaries.
    """

    simple_keys_dict: Dict[str, List[Any]] = defaultdict(list)
    zip_keys_dict: Dict[str, List[Any]] = defaultdict(list)

    for sample in batch:
        for key, value in sample.items():
            if isinstance(value, tuple):
                zip_keys_dict[key].append(value)
            elif torch.is_tensor(value):
                simple_keys_dict[key].append(value)
            else:
                raise ValueError(f"Unsupported value type in sample: {type(value)}")

    for key, value in zip_keys_dict.items():
        batch = list(zip(*value))
        batch = [torch.cat(group) for group in batch]

    for key, value in simple_keys_dict.items():
        zip_keys_dict[key] = torch.cat(value)

    return dict(zip_keys_dict)


def collate_identity(batch):
    """
    Collate function that returns the batch as is.
    Useful when each item in the batch is already a complete batch.
    E.g., when using a BatchSampler that groups indices.
    """
    # batch is a list of batch-returns
    # each element is already the full batch from dataset
    # just return the first (since BatchSampler groups them)
    return batch[0]
