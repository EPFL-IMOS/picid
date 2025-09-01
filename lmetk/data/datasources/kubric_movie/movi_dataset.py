# File:         movi_dataset.py
# Date:         2024/11/12
# Description:  MoviDataset class used for training and model inference

from gzip import GzipFile
from os import path as opath, scandir
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# Dataset configurations
RATIO_TRAIN: float = 0.8
RATIO_VAL: float = 0.1
RATIO_TEST: float = 0.1

# Constants (should not be modified)
DATA_FILENAME: str = "complexes"
COMPRESSED_EXTENSION: str = ".npy.gz"
METADATA_FILENAME: str = "metadata.json"


class MoviDataset(Dataset):
    """Represents a MoVi Combinatorial Complexes dataset"""

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        normalization_file: str,
        nox4: bool,
        include_not_preprocessed: bool = False,
    ):
        """Create a MoviDataset instance

        Parameters
        ----------
        dataset_dir : str
            Root directory of the MoVi dataset
        split : str
            Requested data split; one of [train, val, test]
        normalization_file : str
            Location of the normalization file to use
        nox4 : bool
            True for ablation model
        include_not_preprocessed : bool, optional
            List dataset items which are not pre-processed, by default False
        """
        assert split in ["train", "val", "test"]
        assert normalization_file != ""

        # Private variables
        self.__samples_paths: list[str] = []
        self.__norm = MoviNormalization(normalization_file)
        if include_not_preprocessed:
            self.__filename = METADATA_FILENAME
        else:
            self.__filename = (
                DATA_FILENAME + ("-nox4" if nox4 else "") + COMPRESSED_EXTENSION
            )
        self.__nox4 = nox4

        # Load the data from the specified directory
        self.__read_samples_ids_from_disk(dataset_dir)

        # Assign samples for training, validation, and testing
        rng = np.random.default_rng(seed=0)
        all_indices = np.arange(len(self.__samples_paths)).tolist()
        train_indices = rng.choice(
            all_indices,
            size=int(np.ceil(RATIO_TRAIN * len(self.__samples_paths))),
            replace=False,
        )
        remaining_indices = list(set(all_indices) - set(train_indices))
        val_indices = rng.choice(
            remaining_indices,
            size=int(np.ceil(RATIO_VAL * len(self.__samples_paths))),
            replace=False,
        )
        test_indices: list[int] = list(set(remaining_indices) - set(val_indices))

        # Keep only certain indices, based on the split
        if split == "train":
            self.__samples_paths = [self.__samples_paths[idx] for idx in train_indices]
        elif split == "val":
            self.__samples_paths = [self.__samples_paths[idx] for idx in val_indices]
        else:
            self.__samples_paths = [self.__samples_paths[idx] for idx in test_indices]

        super().__init__()

    @property
    def normalization(self):
        return self.__norm

    @property
    def samples_paths(self) -> list[str]:
        return self.__samples_paths

    def __len__(self) -> int:
        return len(self.__samples_paths)

    def __repr__(self):
        return f"MoviDataset(size={len(self.__samples_paths)}"

    def __getitem__(self, idx: int) -> dict:
        """Get one sample of the dataset"""
        f = GzipFile(opath.join(self.__samples_paths[idx], self.__filename))
        data = np.load(f, allow_pickle=True).item()
        f.close()

        self.__normalize_sample(data)

        return data

    def __normalize_sample(self, data: dict) -> dict:
        data["x0"] = normalize(data["x0"], self.__norm.x0_mean, self.__norm.x0_std)
        data["x1"] = normalize(data["x1"], self.__norm.x1_mean, self.__norm.x1_std)
        data["x2_l"] = normalize_list(
            data["x2_l"], self.__norm.x2_mean, self.__norm.x2_std
        )
        data["x3_l"] = normalize_list(
            data["x3_l"], self.__norm.x3_mean, self.__norm.x3_std
        )
        data["t0"] = normalize(data["t0"], self.__norm.t0_mean, self.__norm.t0_std)
        if not self.__nox4:
            data["x4"] = normalize(data["x4"], self.__norm.x4_mean, self.__norm.x4_std)
            data["t4"] = normalize(data["t4"], self.__norm.t4_mean, self.__norm.t4_std)

        return data

    def __read_samples_ids_from_disk(self, dataset_dir: str) -> None:
        dataset_dir = opath.abspath(dataset_dir)
        if not opath.isdir(dataset_dir):
            raise RuntimeError(f"Dataset directory {dataset_dir} does not exist.")

        samples_dir = [f.name for f in scandir(dataset_dir) if f.is_dir()]
        try:
            samples = set([int(f) for f in samples_dir if len(f) <= 4])
        except Exception as e:
            print(f"ERROR: Dataset directory contains non-digit samples IDs")
            raise e

        self.__samples_paths = [
            opath.join(dataset_dir, str(s))
            for s in samples
            if opath.isfile(opath.join(dataset_dir, str(s), self.__filename))
        ]

        if not len(self.__samples_paths) > 0:
            raise RuntimeError(f"ERROR: Empty dataset {dataset_dir}")


class MoviNormalization(object):
    def __init__(self, normalization_file: str):
        self.__x0_mean: torch.Tensor
        self.__x0_std: torch.Tensor
        self.__x1_mean: torch.Tensor
        self.__x1_std: torch.Tensor
        self.__x2_mean: torch.Tensor
        self.__x2_std: torch.Tensor
        self.__x3_mean: torch.Tensor
        self.__x3_std: torch.Tensor
        self.__x4_mean: torch.Tensor
        self.__x4_std: torch.Tensor
        self.__t0_mean: torch.Tensor
        self.__t0_std: torch.Tensor
        self.__t4_mean: torch.Tensor
        self.__t4_std: torch.Tensor

        # Load the dataset statistics (pre-computed with compute_normalization.py)
        self.__load_normalization(normalization_file)

    @property
    def x0_mean(self) -> torch.Tensor:
        return self.__x0_mean

    @property
    def x0_std(self) -> torch.Tensor:
        return self.__x0_std

    @property
    def x1_mean(self) -> torch.Tensor:
        return self.__x1_mean

    @property
    def x1_std(self) -> torch.Tensor:
        return self.__x1_std

    @property
    def x2_mean(self) -> torch.Tensor:
        return self.__x2_mean

    @property
    def x2_std(self) -> torch.Tensor:
        return self.__x2_std

    @property
    def x3_mean(self) -> torch.Tensor:
        return self.__x3_mean

    @property
    def x3_std(self) -> torch.Tensor:
        return self.__x3_std

    @property
    def x4_mean(self) -> torch.Tensor:
        return self.__x4_mean

    @property
    def x4_std(self) -> torch.Tensor:
        return self.__x4_std

    @property
    def t0_mean(self) -> torch.Tensor:
        return self.__t0_mean

    @property
    def t0_std(self) -> torch.Tensor:
        return self.__t0_std

    @property
    def t4_mean(self) -> torch.Tensor:
        return self.__t4_mean

    @property
    def t4_std(self) -> torch.Tensor:
        return self.__t4_std

    def __load_normalization(self, normalization_file: str) -> None:
        if not opath.isfile(normalization_file):
            raise RuntimeError(
                f"ERROR: Normalization file {normalization_file} does not exist"
            )

        stats: dict = np.load(normalization_file, allow_pickle=True).item()
        self.__x0_mean = torch.tensor(
            stats["x0_mean"], dtype=torch.float32, requires_grad=False
        )
        self.__x0_std = torch.tensor(
            stats["x0_std"], dtype=torch.float32, requires_grad=False
        )
        self.__x1_mean = torch.tensor(
            stats["x1_mean"], dtype=torch.float32, requires_grad=False
        )
        self.__x1_std = torch.tensor(
            stats["x1_std"], dtype=torch.float32, requires_grad=False
        )
        self.__x2_mean = torch.tensor(
            stats["x2_mean"], dtype=torch.float32, requires_grad=False
        )
        self.__x2_std = torch.tensor(
            stats["x2_std"], dtype=torch.float32, requires_grad=False
        )
        self.__x3_mean = torch.tensor(
            stats["x3_mean"], dtype=torch.float32, requires_grad=False
        )
        self.__x3_std = torch.tensor(
            stats["x3_std"], dtype=torch.float32, requires_grad=False
        )
        self.__t0_mean = torch.tensor(
            stats["t0_mean"], dtype=torch.float32, requires_grad=False
        )
        self.__t0_std = torch.tensor(
            stats["t0_std"], dtype=torch.float32, requires_grad=False
        )

        # Special handling of normalization for ablation study without object cells
        self.__x4_mean = (
            torch.tensor(stats["x4_mean"], dtype=torch.float32, requires_grad=False)
            if stats["x4_mean"] is not None
            else None
        )
        self.__x4_std = (
            torch.tensor(stats["x4_std"], dtype=torch.float32, requires_grad=False)
            if stats["x4_std"] is not None
            else None
        )
        self.__t4_mean = (
            torch.tensor(stats["t4_mean"], dtype=torch.float32, requires_grad=False)
            if stats["t4_mean"] is not None
            else None
        )
        self.__t4_std = (
            torch.tensor(stats["t4_std"], dtype=torch.float32, requires_grad=False)
            if stats["t4_std"] is not None
            else None
        )


def normalize_list(
    arrays: list[torch.Tensor], mean: torch.Tensor, std: torch.Tensor
) -> list[torch.Tensor | None]:
    return [(a - mean) / std if a is not None else None for a in arrays]


def normalize(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (t - mean) / std


def denormalize_list(
    arrays: list[torch.Tensor], mean: torch.Tensor, std: torch.Tensor
) -> list[torch.Tensor | None]:
    return [a * std + mean if a is not None else None for a in arrays]


def denormalize(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return t * std + mean


def stack_tensors(
    tensors: list[torch.Tensor], dim: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    return torch.stack(tensors, dim=dim).to(device)


def tensors(
    arrays: list[np.ndarray], device: Optional[torch.device] = None
) -> list[torch.Tensor | None]:
    return [
        torch.from_numpy(a.astype(np.float32)).to(device) if a is not None else None
        for a in arrays
    ]


def movi_dataset_collate_fn(data):
    assert len(data) == 1
    return data[0]
