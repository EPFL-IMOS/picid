"""Hashing helpers for configs and directories."""

from typing import Union, List
import hashlib

import json
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, ListConfig

# def generate_zero_sparse_connectivity(m, n):
#     """Generate a zero sparse connectivity matrix.

#     Parameters
#     ----------
#     m : int
#         Number of rows.
#     n : int
#         Number of columns.

#     Returns
#     -------
#     torch.sparse_coo_tensor
#         Zero sparse connectivity matrix.
#     """
#     return torch.sparse_coo_tensor((m, n)).coalesce()


def ensure_serializable(obj):
    """
    Recursively convert config objects to JSON-serializable types.

    Parameters
    ----------
    obj : Any
        Object to convert.

    Returns
    -------
    Any
        A JSON-serializable representation of ``obj``.
    """
    # Flatten Hydra configs first
    if isinstance(obj, (DictConfig, ListConfig)):
        obj = OmegaConf.to_container(obj, resolve=True)

    if isinstance(obj, dict):
        return {str(k): ensure_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, set):
        # Sort to guarantee deterministic order
        return sorted(ensure_serializable(item) for item in obj)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        # Fallback: force string conversion
        return str(obj)


def hash_config(cfg):
    """
    Make a stable hash from a Hydra/OmegaConf config.

    Parameters
    ----------
    cfg : DictConfig | dict | list
        The configuration object.

    Returns
    -------
    str
        SHA256 hex digest of the config.
    """
    cfg = ensure_serializable(cfg)
    config_str = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()


def hash_directory(directory: str, extensions=None) -> str:
    """
    Compute a hash of files in a directory, optionally filtered by suffix.

    Parameters
    ----------
    directory : str
        Directory to hash recursively.
    extensions : list[str] | None, default=None
        Optional suffix filter such as ``[".py"]``.

    Returns
    -------
    str
        SHA256 hex digest of the selected files.
    """
    sha = hashlib.sha256()
    directory = Path(directory)

    for path in sorted(directory.rglob("*")):  # sorted ensures stable order
        if path.is_file():
            if extensions and path.suffix not in extensions:
                continue
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    sha.update(chunk)
    return sha.hexdigest()


def compute_cache_key(
    config: dict, library_dir: Union[str, List[str], Path], extensions=None
) -> str:
    """
    Combine config and library hashes into one cache key.

    Parameters
    ----------
    config : dict
        Configuration object to hash.
    library_dir : str | list[str] | Path
        Directory or list of directories to hash.
    extensions : list[str] | None, default=None
        Optional suffix filter used when hashing directories.

    Returns
    -------
    str
        Combined cache key.
    """
    config_hash = hash_config(config)

    # Handle both a single directory string and a list of directory strings
    if isinstance(library_dir, str) or isinstance(library_dir, Path):
        # If it's just a string, hash that one directory
        lib_hash = hash_directory(library_dir, extensions)
    elif isinstance(library_dir, list):
        # If it's a list, hash each directory and combine the hashes
        all_lib_hashes = [hash_directory(d, extensions) for d in library_dir]
        # Sort to make the hash independent of the list order
        all_lib_hashes.sort()
        lib_hash = "".join(all_lib_hashes)
    else:
        raise TypeError("library_dir must be a string or a list of strings.")

    combined = hashlib.sha256((config_hash + lib_hash).encode()).hexdigest()
    return combined
