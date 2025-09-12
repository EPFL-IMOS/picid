

import joblib

from pathlib import Path
from lmetk.data.cache.base import BaseCache
import logging
logger = logging.getLogger(__name__)

class StreamCache(BaseCache):
    """
    Cache pipeline where objects are read from a stream,
    transformed, and written to a file.
    """

    def __init__(self, transform_fn, output_dir):
        self.transform_fn = transform_fn
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load(self, name: str):
        path = self.output_dir / f"{name}.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return joblib.load(f)

    def save(self, name: str, data):
        path = self.output_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            joblib.dump(data, f)

    def handle(self, name: str, stream):
        # Different pipeline: consume stream instead of build_fn
        cached = self.load(name)
        if cached is not None:
            return cached

        obj = self.transform_fn(stream)
        self.save(name, obj)
        return obj
