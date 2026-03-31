import logging
from typing import Any, Dict

import numpy as np

# Assuming this import path is correct for your project
from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import (
    ConcatFitAndPerSegmentTransformMixin,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassLabelLookup:
    def __init__(self):
        self.table = {}
        self.counter = 1

    def get(self, key):
        """
        Return the integer label for a key, or ``None`` if missing.

        Parameters
        ----------
        key : Any
            Lookup key.

        Returns
        -------
        int or None
            Assigned label if the key exists.
        """
        return self.table.get(key, None)

    def set(self, key):
        """
        Assign the next integer label to a new key and return it.

        Parameters
        ----------
        key : Any
            Lookup key to register.

        Returns
        -------
        int
            Assigned label.
        """
        if key not in self.table:
            self.table[key] = self.counter
            self.counter += 1
        return self.table[key]


class ConceptClassesBuilder(ConcatFitAndPerSegmentTransformMixin, DenseTransform):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        self.lookup = ClassLabelLookup()
        self.place_key_holder = "n_DS_{key}"

    def fit_data(self, data: NamedTransformInput, metadata: Dict) -> Any:
        for id in np.unique(data["n_DS"]):
            class_key = self.place_key_holder.format(key=int(id))
            self.lookup.set(class_key)

    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> Any:
        concepts = data["concepts"].copy()

        if not np.isin(concepts, [0, 1]).all():
            concepts = np.rint(concepts)
            logger.warning(
                "Non-binary concept was detected and is now rounded to 0 or 1"
            )

        print("sum:", np.unique(np.sum(concepts, axis=1)))
        assert np.isin(
            concepts, [0, 1]
        ).all(), "Non-binary concept is remaining and not supported"
        assert (
            concepts.sum(axis=1) <= 1
        ).all(), "Combined error modes detected and not supported"

        max_index = np.argmax(concepts, axis=1)
        all_zero = np.all(concepts == 0, axis=1)
        classes = max_index + 1
        classes[all_zero] = 0

        # Consider unit id
        class_key = self.place_key_holder.format(key=int(data["n_DS"][0]))
        class_id = self.lookup.get(class_key)
        assert class_id is not None, "Class index was not found in lookup table"
        classes *= class_id

        data["concepts"] = classes[:, None]

        return data

    # Keeping your original __call__ and transform methods for compatibility.
    # They simply delegate to the new transform_data method.
    def __call__(self, data: Any) -> Any:
        return self.transform_data(data, None)
