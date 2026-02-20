import json
from pathlib import Path
import joblib

from lmetk.utils.hash_utils import (
    hash_config,
    ensure_serializable,
    compute_cache_key,
)
import logging

logger = logging.getLogger(__name__)


class FileSystemCache:
    """
    Stage-aware file-based cache with hash validation.
    Uses joblib for serialization, JSON for config, and plain text for hash.

    Layout:
        {cache_dir}/{dataset_name}/{stage}/data.pkl
        {cache_dir}/{dataset_name}/{stage}/meta.json
        {cache_dir}/{dataset_name}/{stage}/hash.txt

    Special case:
        "preprocessed" stage → nested by hash:
        {cache_dir}/{dataset_name}/preprocessed/{hash}/data.pkl
        {cache_dir}/{dataset_name}/preprocessed/{hash}/meta.json
        {cache_dir}/{dataset_name}/preprocessed/{hash}/hash.txt
    """

    def __init__(self):
        pass

    def _paths(self, cache_dir: str, stage: str, cache_key: str = None):
        """
        Return paths for cache artifacts for a given dataset and stage.
        If stage == "preprocessed", requires cache_key and nests under it.
        """
        base_dir = Path(cache_dir).expanduser()

        if stage in ("preprocessed", "boundary"):
            if cache_key is None:
                raise ValueError(f"cache_key required for stage {stage!r}")
            stage_dir = base_dir / stage / cache_key
        else:
            stage_dir = base_dir / stage

        stage_dir.mkdir(parents=True, exist_ok=True)

        return (
            stage_dir / "data.pkl",
            stage_dir / "meta.json",
            stage_dir / "hash.txt",
            stage_dir / "metadata.pkl",
        )

    def load_metadata(self, cache_dir: str, stage: str, cache_key: str = None):
        """
        Load only the config + hash, without loading the data.
        Returns (config, stored_hash) or None if missing.
        """
        _, meta_file, hash_file, _ = self._paths(cache_dir, stage, cache_key)
        if not meta_file.exists() or not hash_file.exists():
            return None

        with open(meta_file, "r") as f:
            config = json.load(f)

        with open(hash_file, "r") as f:
            stored_hash = f.read().strip()

        return config, stored_hash

    def load_data(self, cache_dir: str, stage: str, cache_key: str = None):
        """
        Load only the data.pkl (assumes hash was already validated).
        """
        pkl_file, _, _, metadata_file = self._paths(cache_dir, stage, cache_key)
        if not pkl_file.exists() and not metadata_file.exists():
            logger.info(f"Cache files not found: stage {stage}")
            return None
        return joblib.load(pkl_file), joblib.load(metadata_file)

    def save(
        self,
        cache_dir: str,
        stage: str,
        data,
        metadata,
        config: dict,
        library_dir: str = None,
        extensions=None,
        cache_key: str = None,
    ):
        """Save data, config, and hash separately."""
        if library_dir:
            cache_key = compute_cache_key(config, library_dir, extensions)
        elif cache_key is None:
            cache_key = hash_config(config)

        pkl_file, meta_file, hash_file, metadata_file = self._paths(
            cache_dir, stage, cache_key
        )
        joblib.dump(data, pkl_file)
        joblib.dump(metadata, metadata_file)

        # save config
        with open(meta_file, "w") as f:
            json.dump(ensure_serializable(config), f, indent=2)

        # save hash
        with open(hash_file, "w") as f:
            f.write(cache_key)

    def handle(
        self,
        cache_dir: str,
        stage: str,
        build_fn,
        config: dict,
        library_dir: str = None,
        extensions=None,
    ):
        """
        Load cached data if hash is valid, otherwise rebuild and save.
        """
        cache_key = (
            compute_cache_key(config, library_dir, extensions)
            if library_dir
            else hash_config(config)
        )

        meta = self.load_metadata(
            cache_dir, stage, cache_key if stage == "preprocessed" else None
        )
        if meta:
            _, stored_hash = meta
            if stored_hash == cache_key:
                logger.info(f"✅ hash matches → load the data now, stage={stage}")
                return self.load_data(
                    cache_dir, stage, cache_key if stage == "preprocessed" else None
                )

        # ❌ no cache or invalid hash → rebuild
        data, metadata = build_fn()
        self.save(
            cache_dir, stage, data, metadata, config, library_dir, extensions, cache_key
        )
        return data, metadata
