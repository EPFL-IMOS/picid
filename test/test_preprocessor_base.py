import pytest
import numpy as np

from lmetk.data.preprocessing.base import PreProcessorInterface


class DummyPreprocessor(PreProcessorInterface):
    def get_processed_data_dict(self):
        return {
            "features": {"train": np.array([1]), "val": np.array([2]), "test": np.array([3])},
            "target": {"train": np.array([0]), "val": np.array([0]), "test": np.array([0])},
        }

    def get_meta_data_dict(self):
        return {"meta": "ok"}

    def prepare_data(self) -> None:
        self._prepared = True

    def apply_transforms(self) -> None:
        self._transformed = True

    def pipeline(self):
        self.prepare_data()
        self.apply_transforms()
        return self.get_processed_data_dict()


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

    dp.prepare_data()
    assert dp._prepared is True

    dp.apply_transforms()
    assert dp._transformed is True

    processed = dp.get_processed_data_dict()
    assert isinstance(processed, dict)
    assert "features" in processed and "target" in processed

    for key in ("train", "val", "test"):
        assert isinstance(processed["features"][key], np.ndarray)
        assert isinstance(processed["target"][key], np.ndarray)

    out = dp.pipeline()
    assert out == processed

    meta = dp.get_meta_data_dict()
    assert isinstance(meta, dict)
