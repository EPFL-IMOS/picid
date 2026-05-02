from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from picid.data.data_objects import SplitDatasetContainer, SplitViewPolicy
from picid.exceptions import TransformError
from picid.interface.utils import (
    InterfacePreProcessor,
    ProcessedDatasource,
    get_inverter_for_key_with_name,
    register_data_dim_resolver,
    create_lightning_module,
)
from picid.model.adapters.base import (
    AbstractFeedForwardWrapper,
    AbstractFeedForwardTrainingWrapper,
    AbstractFitPredictWrapper,
)
from picid.transforms.base import DataTransform
from picid.transforms.base.multisource import InverseTransformMixin
from test.transforms.base.conftest import DummyStatelessTransform


def _small_split_container() -> SplitDatasetContainer:
    return SplitDatasetContainer(
        features={
            "train": [np.array([[1.0], [2.0]])],
            "val": [np.array([[3.0]])],
            "test": [np.array([[4.0]])],
        },
        target={
            "train": [np.array([[10.0], [20.0]])],
            "val": [np.array([[30.0]])],
            "test": [np.array([[40.0]])],
        },
    )


def test_interface_apply_transforms_prints_run_style_summary(monkeypatch):
    captured_summaries = []
    monkeypatch.setattr(
        "picid.interface.utils.print_transforms_summary",
        captured_summaries.append,
    )
    data = _small_split_container()
    transform = DataTransform(
        "double_features",
        DummyStatelessTransform(),
        {"apply_to": "features", "assign_to": "features"},
    )

    result = InterfacePreProcessor.apply_transforms(data, [transform], None)

    assert len(captured_summaries) == 1
    summary = captured_summaries[0]
    assert len(summary) == 1
    assert summary[0]["transform_name"] == "double_features"
    assert summary[0]["status"] == "Success"
    assert "dense" in summary[0]["details"]
    assert "features" in summary[0]["inputs"]

    split_result = result.to_split_dict(SplitViewPolicy.KEEP_UNIT_LISTS)
    np.testing.assert_array_equal(
        split_result["train"]["features"][0],
        np.array([[2.0], [4.0]]),
    )


def test_interface_apply_transforms_empty_list_prints_empty_summary(monkeypatch):
    captured_summaries = []
    monkeypatch.setattr(
        "picid.interface.utils.print_transforms_summary",
        captured_summaries.append,
    )
    data = _small_split_container()

    result = InterfacePreProcessor.apply_transforms(data, [], None)

    assert result is data
    assert captured_summaries == [[]]


# ---------------------------------------------------------------------------
# ProcessedDatasource
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProcessedDatasource:
    """Tests for ProcessedDatasource properties and __getattr__ delegation."""

    def _make(self):
        ds = MagicMock()
        ds.custom_attr = "hello"
        return (
            ProcessedDatasource(
                datasource=ds,
                task_mode="regression",
                data_dict={"train": np.zeros((5, 3))},
                meta_data_dict={"n": 42},
            ),
            ds,
        )

    def test_data_dict_property(self):
        pd, _ = self._make()
        assert "train" in pd.data_dict

    def test_meta_data_dict_property(self):
        pd, _ = self._make()
        assert pd.meta_data_dict == {"n": 42}

    def test_datasource_property(self):
        pd, ds = self._make()
        assert pd.datasource is ds

    def test_getattr_delegates_to_datasource(self):
        pd, ds = self._make()
        assert pd.custom_attr == "hello"

    def test_getattr_raises_attribute_error_for_missing(self):
        pd, _ = self._make()
        pd._datasource = MagicMock(spec=[])
        with pytest.raises(AttributeError):
            _ = pd.nonexistent_thing


# ---------------------------------------------------------------------------
# InterfacePreProcessor init, get_meta_data_dict, get_processed_data_dict
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInterfacePreProcessorInit:
    """Tests for InterfacePreProcessor.__init__ and basic methods."""

    def test_init_stores_datasource_and_transforms(self):
        ds = MagicMock()
        t = [MagicMock()]
        pp = InterfacePreProcessor(datasource=ds, transforms=t)
        assert pp.datasource is ds
        assert pp.transforms is t
        assert pp.data is None
        assert pp.meta_data == {}

    def test_get_meta_data_dict_returns_empty_initially(self):
        pp = InterfacePreProcessor(datasource=MagicMock(), transforms=[])
        assert pp.get_meta_data_dict() == {}

    def test_get_processed_data_dict_raises_when_not_preprocessed(self):
        pp = InterfacePreProcessor(datasource=MagicMock(), transforms=[])
        with pytest.raises(RuntimeError, match="pipeline"):
            pp.get_processed_data_dict()

    def test_get_processed_data_dict_returns_data_after_preprocessing(self):
        pp = InterfacePreProcessor(datasource=MagicMock(), transforms=[])
        pp._is_preprocessed = True
        pp.data = _small_split_container()
        result = pp.get_processed_data_dict()
        assert result is pp.data

    def test_get_processed_data_dict_return_splits_on_first_level(self):
        pp = InterfacePreProcessor(datasource=MagicMock(), transforms=[])
        pp._is_preprocessed = True
        pp.data = _small_split_container()
        result = pp.get_processed_data_dict(return_splits_on_first_level=True)
        assert isinstance(result, dict)
        assert "train" in result


# ---------------------------------------------------------------------------
# apply_transforms error paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyTransformsErrors:
    """Tests for apply_transforms ValueError and exception-wrapping branches."""

    def test_non_data_transform_element_raises_value_error(self):
        data = _small_split_container()
        with pytest.raises(ValueError, match="DataTransform"):
            InterfacePreProcessor.apply_transforms(data, ["not_a_transform"], None)

    def test_transform_error_propagates_unchanged(self, monkeypatch):
        monkeypatch.setattr(
            "picid.interface.utils.print_transforms_summary", lambda s: None
        )
        data = _small_split_container()
        t = DataTransform(
            "bad",
            DummyStatelessTransform(),
            {"apply_to": "features", "assign_to": "features"},
        )
        with patch.object(
            DataTransform,
            "forward",
            side_effect=TransformError("explicit", step_id="bad"),
        ):
            with pytest.raises(TransformError, match="explicit"):
                InterfacePreProcessor.apply_transforms(data, [t], None)

    def test_generic_exception_wrapped_in_transform_error(self, monkeypatch):
        monkeypatch.setattr(
            "picid.interface.utils.print_transforms_summary", lambda s: None
        )
        data = _small_split_container()
        t = DataTransform(
            "bad2",
            DummyStatelessTransform(),
            {"apply_to": "features", "assign_to": "features"},
        )
        with patch.object(DataTransform, "forward", side_effect=RuntimeError("raw")):
            with pytest.raises(TransformError):
                InterfacePreProcessor.apply_transforms(data, [t], None)


# ---------------------------------------------------------------------------
# InterfacePreProcessor.pre_process_data, get_meta_data, pipeline
# ---------------------------------------------------------------------------


def _make_mock_datasource():
    """Return a MagicMock datasource that quacks like a CustomSingleSourceLoader."""
    ds = MagicMock()
    ds.get_meta_data.return_value = {"source": "mock"}
    ds.get_data.return_value = _small_split_container()
    ds.task_mode = "regression"
    return ds


@pytest.mark.unit
class TestPreProcessData:
    """Tests for pre_process_data, get_meta_data, and pipeline."""

    def test_pre_process_data_calls_load_split_fetch(self):
        ds = _make_mock_datasource()
        pp = InterfacePreProcessor(datasource=ds, transforms=None)
        meta = pp.pre_process_data()

        ds.load_data.assert_called_once()
        ds.split_data.assert_called_once()
        assert meta == {"source": "mock"}
        assert isinstance(pp.data, SplitDatasetContainer)

    def test_get_meta_data_delegates_to_datasource(self):
        ds = _make_mock_datasource()
        pp = InterfacePreProcessor(datasource=ds, transforms=None)
        meta = pp.get_meta_data(datasource=ds)
        assert meta == {"source": "mock"}

    def test_pipeline_returns_processed_datasource(self, monkeypatch):
        monkeypatch.setattr(
            "picid.interface.utils.print_transforms_summary", lambda s: None
        )
        ds = _make_mock_datasource()
        pp = InterfacePreProcessor(datasource=ds, transforms=None)
        result = pp.pipeline()
        assert isinstance(result, ProcessedDatasource)
        assert result.task_mode == "regression"
        assert result.meta_data_dict == {"source": "mock"}


# ---------------------------------------------------------------------------
# register_data_dim_resolver
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegisterDataDimResolver:
    """Tests for register_data_dim_resolver and its inner infer_data_dim function."""

    @pytest.fixture(autouse=True)
    def _restore_resolver(self):
        """Restore the conftest infer_data_dim resolver after each test."""
        from omegaconf import OmegaConf as _OmegaConf

        yield
        _OmegaConf.register_new_resolver(
            "infer_data_dim", lambda key, dim: 5, replace=True
        )

    def _register_and_resolve(self, data, key, dim):
        """Register resolver then resolve ${infer_data_dim:<key>,<dim>}."""
        from omegaconf import OmegaConf

        register_data_dim_resolver(data)
        cfg = OmegaConf.create({"result": f"${{infer_data_dim:{key},{dim}}}"})
        return OmegaConf.to_container(cfg, resolve=True)["result"]

    def test_numpy_array(self):
        data = {"x": np.zeros((10, 5, 3))}
        result = self._register_and_resolve(data, "x", 1)
        assert result == 5

    def test_torch_tensor(self):
        data = {"t": torch.zeros(4, 7, 2)}
        result = self._register_and_resolve(data, "t", 0)
        assert result == 4

    def test_list_consistent_dims(self):
        data = {"lst": [np.zeros((3, 2)), np.zeros((3, 2))]}
        result = self._register_and_resolve(data, "lst", 1)
        assert result == 2

    def test_list_inconsistent_dims_raises(self):
        data = {"bad": [np.zeros((3, 2)), np.zeros((3, 4))]}
        register_data_dim_resolver(data)
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"result": "${infer_data_dim:bad,1}"})
        with pytest.raises(Exception):
            OmegaConf.to_container(cfg, resolve=True)

    def test_missing_key_raises_key_error(self):
        data = {"x": np.zeros((5, 3))}
        register_data_dim_resolver(data)
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"result": "${infer_data_dim:missing,0}"})
        with pytest.raises(Exception):
            OmegaConf.to_container(cfg, resolve=True)

    def test_unknown_type_raises_value_error(self):
        data = {"u": object()}
        register_data_dim_resolver(data)
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"result": "${infer_data_dim:u,0}"})
        with pytest.raises(Exception):
            OmegaConf.to_container(cfg, resolve=True)

    def test_awkward_array(self):
        import awkward as ak

        data = {"aw": ak.Array([[1, 2, 3], [4, 5, 6]])}
        result = self._register_and_resolve(data, "aw", 0)
        assert result == 2


# ---------------------------------------------------------------------------
# get_inverter_for_key_with_name
# ---------------------------------------------------------------------------


class _InvertibleTransform(InverseTransformMixin, DummyStatelessTransform):
    """Minimal invertible transform for testing."""

    def inverse_transform(self, data):
        return data, {}

    def inverse_forward(self, data):
        return data, {}

    @property
    def assign_to(self):
        return self._assign_to

    def __init__(self, assign_to="features"):
        super().__init__()
        self._assign_to = assign_to


@pytest.mark.unit
class TestGetInverterForKeyWithName:
    """Tests for get_inverter_for_key_with_name."""

    def _make_dt(self, name, assign_to="features"):
        t = _InvertibleTransform(assign_to=assign_to)
        dt = DataTransform(name, t, {"apply_to": assign_to, "assign_to": assign_to})
        return dt

    def test_invalid_which_raises_value_error(self):
        with pytest.raises(ValueError, match="first.*last"):
            get_inverter_for_key_with_name([], "features", which="middle")

    def test_no_invertible_transforms_returns_none_none(self):
        non_inv = DataTransform(
            "plain", DummyStatelessTransform(), {"apply_to": "features"}
        )
        result = get_inverter_for_key_with_name([non_inv], "features")
        assert result == (None, None)

    def test_last_invertible_transform_returned_by_default(self):
        dt1 = self._make_dt("first", "features")
        dt2 = self._make_dt("second", "features")
        inv, name = get_inverter_for_key_with_name([dt1, dt2], "features", which="last")
        assert name == "second"

    def test_first_invertible_transform_returned(self):
        dt1 = self._make_dt("first", "features")
        dt2 = self._make_dt("second", "features")
        inv, name = get_inverter_for_key_with_name(
            [dt1, dt2], "features", which="first"
        )
        assert name == "first"

    def test_key_mismatch_returns_none_none(self):
        dt = self._make_dt("inv", "target")
        inv, name = get_inverter_for_key_with_name([dt], "features")
        assert (inv, name) == (None, None)


# ---------------------------------------------------------------------------
# create_lightning_module backbone routing
# ---------------------------------------------------------------------------


class _MinFFW(AbstractFeedForwardWrapper):
    def forward(self, batch):
        return {}


class _MinFFTW(AbstractFeedForwardTrainingWrapper):
    def forward(self, batch):
        return {}


class _MinFPW(AbstractFitPredictWrapper):
    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

    def serialize_model(self, task_id):
        pass

    def load_model(self, task_id):
        pass

    @property
    def allows_multi_target(self):
        return False


def _make_backbone(cls):
    """Create a bare instance of a wrapper subclass without calling __init__."""
    inst = object.__new__(cls)
    if issubclass(cls, torch.nn.Module):
        torch.nn.Module.__init__(inst)
    return inst


def _minimal_cfg(with_scheduler=False):
    from omegaconf import OmegaConf

    cfg_dict = {
        "optimization": {
            "optimizer": {"_target_": "torch.optim.AdamW"},
        },
        "datasource": {"data_name": "tabular"},
        "model": {},
    }
    if with_scheduler:
        cfg_dict["optimization"]["scheduler"] = {
            "_target_": "torch.optim.lr_scheduler.StepLR"
        }
    return OmegaConf.create(cfg_dict)


@pytest.mark.unit
class TestCreateLightningModuleRouting:
    """Tests for create_lightning_module backbone-type routing (lines 180-268)."""

    @patch("picid.interface.utils.ConstantLossLightningModule")
    @patch("picid.interface.utils.hydra")
    def test_feedforward_wrapper_creates_constant_loss_module(
        self, mock_hydra, mock_clm
    ):
        mock_hydra.utils.instantiate.return_value = MagicMock()
        mock_clm.return_value = sentinel = MagicMock()

        backbone = _make_backbone(_MinFFW)
        result = create_lightning_module(
            _minimal_cfg(), MagicMock(), [], MagicMock(), backbone
        )

        mock_clm.assert_called_once()
        assert result is sentinel

    @patch("picid.interface.utils.TrainingLightningModule")
    @patch("picid.interface.utils.hydra")
    def test_feedforward_training_wrapper_creates_training_module(
        self, mock_hydra, mock_tlm
    ):
        mock_hydra.utils.instantiate.return_value = MagicMock()
        mock_tlm.return_value = sentinel = MagicMock()

        backbone = _make_backbone(_MinFFTW)
        result = create_lightning_module(
            _minimal_cfg(), MagicMock(), [], MagicMock(), backbone
        )

        mock_tlm.assert_called_once()
        assert result is sentinel

    @patch("picid.interface.utils.FitPredictWrapperLightningModule")
    @patch("picid.interface.utils.hydra")
    def test_fit_predict_wrapper_creates_fit_predict_module(self, mock_hydra, mock_fpm):
        mock_hydra.utils.instantiate.return_value = MagicMock()
        mock_fpm.return_value = sentinel = MagicMock()

        backbone = _make_backbone(_MinFPW)
        result = create_lightning_module(
            _minimal_cfg(), MagicMock(), [], MagicMock(), backbone
        )

        mock_fpm.assert_called_once()
        assert result is sentinel

    @patch("picid.interface.utils.hydra")
    def test_unknown_backbone_type_raises_value_error(self, mock_hydra):
        mock_hydra.utils.instantiate.return_value = MagicMock()

        backbone = MagicMock(spec=[])  # not a recognized wrapper
        with pytest.raises(ValueError, match="Unsupported"):
            create_lightning_module(
                _minimal_cfg(), MagicMock(), [], MagicMock(), backbone
            )

    @patch("picid.interface.utils.ConstantLossLightningModule")
    @patch("picid.interface.utils.hydra")
    def test_scheduler_factory_created_when_cfg_has_scheduler(
        self, mock_hydra, mock_clm
    ):
        mock_hydra.utils.instantiate.return_value = MagicMock()
        mock_clm.return_value = MagicMock()

        backbone = _make_backbone(_MinFFW)
        create_lightning_module(
            _minimal_cfg(with_scheduler=True), MagicMock(), [], MagicMock(), backbone
        )

        # instantiate called twice: once for optimizer, once for scheduler
        assert mock_hydra.utils.instantiate.call_count == 2

    @patch("picid.interface.utils.ConstantLossLightningModule")
    @patch("picid.interface.utils.hydra")
    def test_backbone_none_with_lightning_model_returns_directly(
        self, mock_hydra, mock_clm
    ):
        """backbone=None + metadata.lightning_model=True → model from hydra, returned as-is."""
        from omegaconf import OmegaConf

        sentinel = MagicMock()
        mock_hydra.utils.instantiate.return_value = sentinel

        cfg = OmegaConf.create(
            {
                "optimization": {"optimizer": {"_target_": "torch.optim.AdamW"}},
                "datasource": {"data_name": "tabular"},
                "model": {"metadata": {"lightning_model": True}},
            }
        )
        result = create_lightning_module(
            cfg, MagicMock(), [], MagicMock(), backbone=None
        )

        assert result is sentinel
        mock_clm.assert_not_called()

    @patch("picid.interface.utils.ConstantLossLightningModule")
    @patch("picid.interface.utils.hydra")
    def test_backbone_none_with_load_path_loads_state_dict(self, mock_hydra, mock_clm):
        """backbone=None + cfg.model.load_path → backbone.backbone.load_state_dict called."""
        from omegaconf import OmegaConf

        fake_backbone = _make_backbone(_MinFFW)
        fake_backbone.backbone = MagicMock()
        mock_clm.return_value = MagicMock()

        def _instantiate(cfg, **kwargs):
            if hasattr(cfg, "optimizer") or (
                hasattr(cfg, "_target_") and "optim" in cfg._target_
            ):
                return MagicMock()
            return fake_backbone

        mock_hydra.utils.instantiate.side_effect = _instantiate

        cfg = OmegaConf.create(
            {
                "optimization": {"optimizer": {"_target_": "torch.optim.AdamW"}},
                "datasource": {"data_name": "tabular"},
                "model": {"load_path": "/fake/path.pt"},
            }
        )
        with patch("picid.interface.utils.torch.load", return_value={}):
            create_lightning_module(cfg, MagicMock(), [], MagicMock(), backbone=None)

        fake_backbone.backbone.load_state_dict.assert_called_once_with({})

    @patch("picid.interface.utils.ConstantLossLightningModule")
    @patch("picid.interface.utils.hydra")
    def test_backbone_none_railway_datasource_adds_dataloaders(
        self, mock_hydra, mock_clm
    ):
        """backbone=None + datasource.data_name='railway' → dataloaders passed to instantiate."""
        from omegaconf import OmegaConf

        mock_clm.return_value = MagicMock()
        fake_backbone = _make_backbone(_MinFFW)
        mock_hydra.utils.instantiate.return_value = fake_backbone

        cfg = OmegaConf.create(
            {
                "optimization": {"optimizer": {"_target_": "torch.optim.AdamW"}},
                "datasource": {"data_name": "railway"},
                "model": {},
            }
        )
        dm = MagicMock()
        create_lightning_module(cfg, dm, [], MagicMock(), backbone=None)

        dm.train_dataloader.assert_called_once()
        dm.val_dataloader.assert_called_once()
