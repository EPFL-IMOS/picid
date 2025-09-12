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
    Stage-aware file-based cache with hash validation.
    Uses joblib for serialization and stores metadata in JSON.

    Each stage (e.g., "loaded_and_splitted_data", "preprocessed_data") has its own subfolder.
    """

    def __init__(self):
        # No global cache_dir or library_dir
        pass

    def _paths(self, cache_dir: str, dataset_name: str, stage: str):
        """
        Return paths for cache artifacts for a given dataset and stage.
        Example layout:
            {cache_dir}/{dataset_name}/{stage}/data.pkl
            {cache_dir}/{dataset_name}/{stage}/meta.json
        """
        stage_dir = Path(cache_dir).expanduser() / dataset_name / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        pkl_file = stage_dir / "data.pkl"
        meta_file = stage_dir / "meta.json"
        return pkl_file, meta_file

    def load(self, cache_dir: str, dataset_name: str, stage: str):
        """Load cached data and metadata if available."""
        pkl_file, meta_file = self._paths(cache_dir, dataset_name, stage)
        if not pkl_file.exists() or not meta_file.exists():
            return None

        with open(meta_file, "r") as f:
            metadata = json.load(f)

        data = joblib.load(pkl_file)
        return data, metadata

    def save(self, cache_dir: str, dataset_name: str, stage: str, data, config: dict, library_dir: str = None, extensions=None):
        """Save data and metadata with hash validation."""
        pkl_file, meta_file = self._paths(cache_dir, dataset_name, stage)
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

    def handle(self, cache_dir: str, dataset_name: str, stage: str, build_fn, config: dict, library_dir: str = None, extensions=None):
        """
        Load cached data if valid, otherwise rebuild via build_fn and save.
        """
        expected_key = (
            compute_cache_key(config, library_dir, extensions)
            if library_dir
            else hash_config(config)
        )

        cached = self.load(cache_dir, dataset_name, stage)
        if cached:
            data, metadata = cached
            if metadata["hash"] == expected_key:
                return data  # ✅ cache is valid

        # cache invalid → rebuild
        data = build_fn()
        self.save(cache_dir, dataset_name, stage, data, config, library_dir, extensions)
        return data
