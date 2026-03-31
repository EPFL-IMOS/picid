import json
from pathlib import Path
import joblib

from picid.utils.hash_utils import (
    hash_config,
    ensure_serializable,
    compute_cache_key,
)
import logging

logger = logging.getLogger(__name__)


class FileSystemCache:
    """
    Stage-aware file cache with hash validation.

    The cache stores data, metadata, and a hash separately so preprocessing can
    validate whether the serialized artifacts still match the current code and
    configuration.
    """

    def __init__(self):
        pass

    def _paths(self, cache_dir: str, stage: str, cache_key: str = None):
        """
        Return cache artifact paths for the given stage.

        Parameters
        ----------
        cache_dir : str
            Root cache directory.
        stage : str
            Cache stage name.
        cache_key : str | None, optional
            Nested cache key used by ``preprocessed`` and ``boundary`` stages.

        Returns
        -------
        tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]
            Paths for data, metadata JSON, hash text, and metadata pickle.
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
        Load only the config and hash without loading the data.

        Parameters
        ----------
        cache_dir : str
            Root cache directory.
        stage : str
            Cache stage name.
        cache_key : str | None, optional
            Nested cache key used by ``preprocessed`` and ``boundary`` stages.

        Returns
        -------
        tuple[dict, str] | None
            Loaded configuration and stored hash, or ``None`` when missing.
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
        Load the cached data and metadata payloads.

        Parameters
        ----------
        cache_dir : str
            Root cache directory.
        stage : str
            Cache stage name.
        cache_key : str | None, optional
            Nested cache key used by ``preprocessed`` and ``boundary`` stages.

        Returns
        -------
        tuple[Any, Any] | None
            Cached data and metadata objects, or ``None`` when absent.
        """
        pkl_file, _, _, metadata_file = self._paths(cache_dir, stage, cache_key)
        if not pkl_file.exists() and not metadata_file.exists():
            logger.info(f"Cache files not found: stage {stage}")
            return None
        return joblib.load(pkl_file), joblib.load(metadata_file)

    def write_meta(
        self,
        cache_dir: str,
        stage: str,
        config: dict,
        cache_key: str = None,
        library_dir=None,
        extensions=None,
    ) -> str:
        """
        Write meta_pending.json before data computation starts.

        Acts as a tombstone so that if the process crashes during data creation,
        the file exists on disk to record what was being computed. A subsequent
        run will find no meta.json (only meta_pending.json) and rebuild normally.
        Once save() completes it writes the canonical meta.json alongside the data.

        Returns the cache_key used (computed if not provided).
        """
        if library_dir:
            cache_key = compute_cache_key(config, library_dir, extensions)
        elif cache_key is None:
            cache_key = hash_config(config)
        _, meta_file, _, _ = self._paths(cache_dir, stage, cache_key)
        pending_file = meta_file.with_name("meta_pending.json")
        with open(pending_file, "w") as f:
            json.dump(ensure_serializable(config), f, indent=2)
        return cache_key

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
        """
        Save data, config, and hash separately.

        Parameters
        ----------
        cache_dir : str
            Root cache directory.
        stage : str
            Cache stage name.
        data : Any
            Serialized data payload.
        metadata : Any
            Serialized metadata payload.
        config : dict
            Configuration used to derive the cache hash.
        library_dir : str | None, optional
            Source library path used to derive a deterministic cache key.
        extensions : iterable[str] | None, optional
            File extensions included in the cache key fingerprint.
        cache_key : str | None, optional
            Explicit cache key override.
        """
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
        Load cached data when valid, otherwise rebuild and save.

        Parameters
        ----------
        cache_dir : str
            Root cache directory.
        stage : str
            Cache stage name.
        build_fn : callable
            Function that rebuilds the data and metadata payloads.
        config : dict
            Configuration used to derive the cache hash.
        library_dir : str | None, optional
            Source library path used to derive a deterministic cache key.
        extensions : iterable[str] | None, optional
            File extensions included in the cache key fingerprint.

        Returns
        -------
        tuple[Any, Any]
            Cached or freshly built data and metadata payloads.
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
        self.write_meta(cache_dir, stage, config, cache_key=cache_key)
        data, metadata = build_fn()
        self.save(
            cache_dir, stage, data, metadata, config, library_dir, extensions, cache_key
        )
        return data, metadata
