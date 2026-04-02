"""Collation helpers for batched model inputs."""

import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any


def collate_key_value_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate a list of dictionaries into a single dictionary.

    Parameters
    ----------
    batch : list[dict[str, Any]]
        Samples to merge into a single batch.

    Returns
    -------
    dict[str, Any]
        Collated batch keyed by sample field name.
    """

    simple_keys_dict: Dict[str, List[Any]] = defaultdict(list)
    zip_keys_dict: Dict[str, List[Any]] = defaultdict(list)

    for sample in batch:
        for key, value in sample.items():
            if isinstance(value, tuple):
                zip_keys_dict[key].append(value)
            elif torch.is_tensor(value):
                simple_keys_dict[key].append(value)
            elif isinstance(value, np.ndarray):
                simple_keys_dict[key].append(
                    torch.from_numpy(np.ascontiguousarray(value))
                )
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
    Return the first already-batched item unchanged.

    Parameters
    ----------
    batch : list[Any]
        Already batched samples produced by a batch sampler.

    Returns
    -------
    Any
        First batch element.
    """
    # batch is a list of batch-returns
    # each element is already the full batch from dataset
    # just return the first (since BatchSampler groups them)
    return batch[0]
