import json
from pathlib import Path
import joblib

from lmetk.utils.hash_utils import (
    hash_config,
    ensure_serializable,
    compute_cache_key,
)


class FileSystemCache:
    """
    File-based cache with hash validation.
    Uses joblib for serialization and stores metadata in JSON.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, name: str):
        """
        Return paths for cache artifacts.
        - data.pkl   : cached data
        - meta.json  : metadata (config + hash)
        """
        pkl_file = self.cache_dir / f"{name}.pkl"
        meta_file = self.cache_dir / f"{name}.json"
        return pkl_file, meta_file

    def load(self, name: str):
        """Load cached data and metadata if available."""
        pkl_file, meta_file = self._paths(name)
        if not pkl_file.exists() or not meta_file.exists():
            return None

        # Load metadata
        with open(meta_file, "r") as f:
            metadata = json.load(f)

        # Load cached data
        data = joblib.load(pkl_file)
        return data, metadata

    def save(self, name: str, data, config: dict, library_dir: str = None, extensions=None):
        """Save data and metadata with hash validation."""
        pkl_file, meta_file = self._paths(name)
        joblib.dump(data, pkl_file)

        if library_dir:
            cache_key = compute_cache_key(config, library_dir, extensions)
        else:
            cache_key = hash_config(config)

        metadata = {
            "hash": cache_key,
            "config": ensure_serializable(config),
        }
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def handle(self, name: str, build_fn, config: dict, library_dir: str = None, extensions=None):
        """
        Load cached data if valid, otherwise rebuild via build_fn and save.
        """
        expected_key = (
            compute_cache_key(config, library_dir, extensions)
            if library_dir
            else hash_config(config)
        )

        cached = self.load(name)
        if cached:
            data, metadata = cached
            if metadata["hash"] == expected_key:
                return data  # ✅ cache is valid

        # cache invalid → rebuild
        data = build_fn()
        self.save(name, data, config, library_dir, extensions)
        return data
