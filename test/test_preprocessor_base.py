import pytest
import numpy as np

from picid.data.preprocessing.base import PreProcessorInterface
from picid.data.data_objects import SplitDatasetContainer, SplitViewPolicy


class DummyPreprocessor(PreProcessorInterface):
    def get_processed_data_container(self):
        return SplitDatasetContainer(
            features={
                "train": [np.array([1])],
                "val": [np.array([2])],
                "test": [np.array([3])],
            },
            target={
                "train": [np.array([0])],
                "val": [np.array([0])],
                "test": [np.array([0])],
            },
        )

    def get_processed_split_dict(
        self,
        view_policy: SplitViewPolicy = SplitViewPolicy.KEEP_UNIT_LISTS,
    ):
        return self.get_processed_data_container().to_split_dict(view_policy)

    def get_meta_data_dict(self):
        return {"meta": "ok"}

    def fetch_data(self) -> None:
        self._prepared = True

    def apply_transforms(self) -> None:
        self._transformed = True

    def pipeline(self):
        self.fetch_data()
        self.apply_transforms()
        return self.get_processed_data_container()


def test_abstract_preprocessor_cannot_instantiate():
    # PreProcessorInterface is abstract; direct instantiation should fail
    with pytest.raises(TypeError):
        PreProcessorInterface(datasource=None, transforms=None)


def test_dummy_preprocessor_basic_flow():
    dp = DummyPreprocessor(datasource="ds", transforms="t")
    assert dp.datasource == "ds"
    assert dp.transforms == "t"

    # flags should be set by the concrete implementations
    assert not getattr(dp, "_prepared", False)
    assert not getattr(dp, "_transformed", False)

    dp.fetch_data()
    assert dp._prepared is True

    dp.apply_transforms()
    assert dp._transformed is True

    processed = dp.get_processed_split_dict()
    assert isinstance(processed, dict)
    assert "train" in processed and "val" in processed and "test" in processed

    for key in ("train", "val", "test"):
        assert isinstance(processed[key]["features"], list)
        assert isinstance(processed[key]["target"], list)

    out = dp.pipeline()
    assert isinstance(out, SplitDatasetContainer)

    meta = dp.get_meta_data_dict()
    assert isinstance(meta, dict)
