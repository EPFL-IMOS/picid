"""Abstract class that provides an interface to handle the network output."""

import sys
import logging
from abc import ABC, abstractmethod

from einops import rearrange
import numpy as np
from omegaconf import OmegaConf
import torch
from lightning_fabric.utilities.data import AttributeDict
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AbstractFitPredictWrapper(ABC):
    """
    AbstractFitPredictWrapper(ABC)
    An abstract base class that provides a unified interface for
    handling models that follow the scikit-learn API.
    This wrapper ensures that the backbone model
    implements both `fit` and `predict` methods, and provides batch
    prediction capabilities with optional yield strategy.

    backbone : object
        The backbone model, expected to implement `fit` and `predict`
        methods (e.g., a scikit-learn estimator).
    yield_strategy : bool, optional (default=False)
        If True, enables batch-wise prediction to handle large datasets efficiently.
    yield_batch_size : int, optional (default=128)
        The batch size to use when `yield_strategy` is enabled.
    **kwargs : dict
        Additional keyword arguments to be stored in an AttributeDict.
    Methods
    -------
    fit(X, y)
        Fits the backbone model to the provided data.
    predict(X)
        Predicts outputs for the given input data, optionally in batches
        if `yield_strategy` is enabled. Ensures output is a 2D torch.Tensor.
    serialize_model(task_id)
        Abstract method to serialize the model to a file, to be implemented by subclasses.
    load_model(task_id)
        Abstract method to load the model from a file, to be implemented by subclasses.
    Properties
    allows_multi_target : bool
        Abstract property indicating whether the model supports multi-target prediction.
    Raises
    ------
    ValueError
        If the backbone model does not implement both `fit` and `predict` methods.
    TypeError
        If the predicted output is not convertible to a torch.Tensor or numpy.ndarray.
    AssertionError
        If the predicted output is not 2-dimensional.
    """

    def __init__(
        self,
        backbone,
        yield_strategy: bool = False,
        yield_batch_size: int = 128,
        profile_fit_cfg: OmegaConf = None,
        profile_predict_cfg: OmegaConf = None,
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.kwargs = AttributeDict(kwargs)
        self.yield_strategy = yield_strategy
        self.yield_batch_size = yield_batch_size
        self.profile_fit_cfg = profile_fit_cfg
        self.profile_predict_cfg = profile_predict_cfg
        self.profile_started_flag = False

        # Check if backbone supports fit predict
        if not hasattr(self.backbone, "fit") or not hasattr(self.backbone, "predict"):
            raise ValueError("Backbone model must implement fit and predict methods.")

        if self.profile_fit_cfg or self.profile_predict_cfg:
            assert not (
                self.profile_fit_cfg and self.profile_predict_cfg
            ), "Only one of profile_fit or profile_predict can be set."

    def fit(self, X, y):
        if self.profile_fit_cfg is not None:
            from torch.profiler import record_function

            if not self.profile_started_flag:

                self._setup_profiler(self.profile_fit_cfg)
                self.profile_started_flag = True
            with record_function("dataloader_iteration"):
                self.backbone.fit(X.cpu(), y.cpu())
            self.profiler.step()
            self._check_stop_profiler(self.profile_fit_cfg)
        else:
            self.backbone.fit(X.cpu(), y.cpu())

    def predict(self, X):

        X = X.cpu()
        if self.yield_strategy:
            batch_size = self.yield_batch_size
            n_samples = X.shape[0]
            outputs_list = []
            for start in tqdm(
                range(0, n_samples, batch_size),
                desc=f"Predicting in batches of size {batch_size}",
            ):
                end = min(start + batch_size, n_samples)
                batch_X = X[start:end]

                if self.profile_predict_cfg is not None:
                    from torch.profiler import record_function

                    if not self.profile_started_flag:
                        self._setup_profiler(self.profile_predict_cfg)
                        self.profile_started_flag = True
                    with record_function("dataloader_iteration"):
                        batch_outputs = self.backbone.predict(batch_X)
                    self.profiler.step()
                    self._check_stop_profiler(self.profile_predict_cfg)
                else:
                    batch_outputs = self.backbone.predict(batch_X)

                outputs_list.append(batch_outputs)
            outputs = np.concatenate(outputs_list, axis=0)
            y = torch.from_numpy(outputs)
        else:
            y = self.backbone.predict(X)

        if not isinstance(y, torch.Tensor):
            if hasattr(y, "__array__"):
                y = torch.from_numpy(y)
            else:
                raise TypeError(
                    "Predicted output must be a torch.Tensor or numpy.ndarray."
                )

        if y.ndim == 1:
            y = rearrange(y, "a -> a 1")
        assert y.ndim == 2, "Predicted output must be 2-dimensional."
        return y

    @abstractmethod
    def serialize_model(self, task_id):
        """Serialize the model to a file."""
        pass

    @abstractmethod
    def load_model(self, task_id):
        """Load the model from a file."""
        pass

    @property
    @abstractmethod
    def allows_multi_target(self) -> bool:
        """Indicates whether the model supports multi-target prediction."""
        pass

    def _setup_profiler(self, cfg: OmegaConf):
        from torch.profiler import (
            ProfilerActivity,
            profile,
            schedule,
        )

        sched = schedule(
            wait=cfg.wait,
            warmup=cfg.warmup,
            active=cfg.active,
            repeat=cfg.repeat,
        )

        acts = [getattr(ProfilerActivity, a) for a in cfg.activities]
        profiler = profile(
            activities=acts,
            schedule=sched,
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )

        self.sched = sched
        self.profiler = profiler

        profiler.start()

    def _check_stop_profiler(self, cfg):
        state = self.sched(self.profiler.step_num)
        from torch.profiler import ProfilerAction

        if state == ProfilerAction.NONE:
            self.profiler.stop()
            logger.warning(
                "Profiler has finished, terminating program. Printing results..."
            )
            logger.warning(
                "\n%s",
                self.profiler.key_averages().table(
                    sort_by="self_cpu_time_total", row_limit=10
                ),
            )
            self.profiler.export_chrome_trace(cfg.trace_path)
            logger.warning(f"Profiler trace written to {cfg.trace_path}")
            sys.exit(0)  # terminate program after profiler is done


class AbstractFeedForwardWrapper(ABC, torch.nn.Module):
    """Abstract class that provides an interface to handle the network output.

    Parameters
    ----------
    backbone : torch.nn.Module
        Backbone model.
    **kwargs : dict
        Additional arguments for the class. It should contain the following keys:
        - out_channels (int): Number of output channels.
        - num_cell_dimensions (int): Number of cell dimensions.
    """

    def __init__(self, backbone, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.kwargs = AttributeDict(kwargs)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(backbone={self.backbone}, "
            f"out_channels={self.backbone.out_channels}, "
            f"dimensions={self.dimensions},"
            f"residual_connections={self.residual_connections})"
        )

    def __call__(self, batch):
        """Forward pass for the model.

        This method calls the forward method and adds the residual connection.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the model output.
        """
        model_out = self.forward(batch)

        return model_out

    @abstractmethod
    def forward(self, batch):
        """Forward pass for the model.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        """


class AbstractFeedForwardTrainingWrapper(ABC, torch.nn.Module):
    """Abstract class that provides an interface to handle the network output.

    Parameters
    ----------
    backbone : torch.nn.Module
        Backbone model.
    **kwargs : dict
        Additional arguments for the class. It should contain the following keys:
        - out_channels (int): Number of output channels.
        - num_cell_dimensions (int): Number of cell dimensions.
    """

    def __init__(self, backbone, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.kwargs = AttributeDict(kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(backbone={self.backbone}"

    def __call__(self, batch):
        """Forward pass for the model.

        This method calls the forward method and adds the residual connection.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the model output.
        """
        model_out = self.forward(batch)

        return model_out

    @abstractmethod
    def forward(self, batch):
        """Forward pass for the model.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        """
