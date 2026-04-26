"""Shared adapter base classes for feed-forward and fit/predict models."""

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
    Unified adapter for sklearn-style fit/predict backbones.

    Parameters
    ----------
    backbone : object
        Model instance or factory expected to expose ``fit`` and ``predict``.
    yield_strategy : bool, default=False
        Whether to run prediction in batches and concatenate the outputs.
    yield_batch_size : int, default=128
        Batch size used when ``yield_strategy`` is enabled.
    profile_fit_cfg : OmegaConf, optional
        Profiler configuration for the fit path.
    profile_predict_cfg : OmegaConf, optional
        Profiler configuration for the predict path.
    reinit_on_fit : bool, default=False
        Whether to rebuild the backbone from a stored factory before each fit.
    verbose : bool, default=False
        Whether to wrap batched prediction in a progress bar.
    **kwargs : dict
        Additional keyword arguments stored on ``self.kwargs``.

    Raises
    ------
    ValueError
        If the backbone model does not implement both ``fit`` and ``predict``.
    TypeError
        If the predicted output is not convertible to a torch tensor.
    AssertionError
        If the predicted output is not two-dimensional.
    """

    def __init__(
        self,
        backbone,
        yield_strategy: bool = False,
        yield_batch_size: int = 128,
        profile_fit_cfg: OmegaConf = None,
        profile_predict_cfg: OmegaConf = None,
        reinit_on_fit: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Store the backbone and shared runtime configuration.

        Parameters
        ----------
        backbone : object
            Model instance or factory expected to expose ``fit`` and ``predict``.
        yield_strategy : bool, default=False
            Whether to run prediction in batches.
        yield_batch_size : int, default=128
            Batch size used for batched prediction.
        profile_fit_cfg : OmegaConf, optional
            Profiler configuration for the fit path.
        profile_predict_cfg : OmegaConf, optional
            Profiler configuration for the predict path.
        reinit_on_fit : bool, default=False
            Whether to rebuild the backbone from a factory before fitting.
        verbose : bool, default=False
            Whether to show a progress bar during batched prediction.
        **kwargs : dict
            Extra wrapper metadata stored in ``self.kwargs``.
        """
        super().__init__()

        # Accept either a model instance or a callable factory (e.g., functools.partial)
        # If a callable is provided we will instantiate it immediately and keep the
        # factory for optional re-initialization before fit.
        if callable(backbone) and not (
            hasattr(backbone, "fit") and hasattr(backbone, "predict")
        ):
            assert (
                reinit_on_fit
            ), "When providing a callable backbone, reinit_on_fit must be True."

            # treat `backbone` as a factory that returns a model instance when called
            self.backbone_factory = backbone
        else:
            assert not callable(
                backbone
            ), "We can not reinitialize on fit when providing a model instance."

            self.backbone_factory = None
            self.backbone = backbone
            # Check if backbone supports fit/predict
            if not hasattr(self.backbone, "fit") or not hasattr(
                self.backbone, "predict"
            ):
                raise ValueError(
                    "Backbone model must implement fit and predict methods."
                )

        self.kwargs = AttributeDict(kwargs)
        self.yield_strategy = yield_strategy
        self.yield_batch_size = yield_batch_size
        self.profile_fit_cfg = profile_fit_cfg
        self.profile_predict_cfg = profile_predict_cfg
        self.profile_started_flag = False
        self.reinit_on_fit = reinit_on_fit
        self.verbose = verbose

        if self.profile_fit_cfg or self.profile_predict_cfg:
            assert not (
                self.profile_fit_cfg and self.profile_predict_cfg
            ), "Only one of profile_fit or profile_predict can be set."

    def _reinit_backbone(self):
        """Re-instantiate the backbone from the captured factory."""
        if self.backbone_factory is None:
            raise RuntimeError(
                "No backbone factory available to re-initialize the model."
            )

        self.backbone = self.backbone_factory()
        # sanity check after re-init
        if not hasattr(self.backbone, "fit") or not hasattr(self.backbone, "predict"):
            raise ValueError(
                "Re-initialized backbone must implement fit and predict methods."
            )

    def _call_fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Fit the wrapped backbone, optionally after reinitialization.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor moved to CPU before fitting.
        y : torch.Tensor
            Target tensor moved to CPU before fitting.
        """
        if self.reinit_on_fit:
            self._reinit_backbone()

        self.backbone.fit(X, y)

    def _call_predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict with the wrapped backbone.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor moved to CPU before prediction.

        Returns
        -------
        torch.Tensor
            Raw prediction tensor returned by the wrapped model.
        """
        return self.backbone.predict(X)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Fit the backbone model on CPU tensors.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor used for fitting.
        y : torch.Tensor
            Target tensor used for fitting.
        """
        if self.profile_fit_cfg is not None:
            from torch.profiler import record_function

            if not self.profile_started_flag:
                self._setup_profiler(self.profile_fit_cfg)
                self.profile_started_flag = True
            with record_function("dataloader_iteration"):
                self._call_fit(X.cpu(), y.cpu())
            self.profiler.step()
            self._check_stop_profiler(self.profile_fit_cfg)
        else:
            self._call_fit(X.cpu(), y.cpu())

    def _iterate_batches(self, X: torch.Tensor):
        """
        Yield CPU batches for the optional batched prediction path.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor that will be split into batches.

        Yields
        ------
        torch.Tensor
            A CPU batch with shape ``(batch_size, ...)``.
        """
        X = X.cpu()  # preprocessing step
        batch_size = self.yield_batch_size
        n_samples = X.shape[0]

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_X = X[start:end]
            yield batch_X

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict outputs for the given input tensor.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor used for prediction.

        Returns
        -------
        torch.Tensor
            Two-dimensional tensor containing the predictions.
        """
        X = X.cpu()
        if self.yield_strategy:
            outputs_list = []

            if self.verbose:
                iterator = tqdm(
                    self._iterate_batches(X),
                    desc=f"Predicting in batches of size {self.yield_batch_size}",
                )
            else:
                iterator = self._iterate_batches(X)

            for batch_X in iterator:
                if self.profile_predict_cfg is not None:
                    from torch.profiler import record_function

                    if not self.profile_started_flag:
                        self._setup_profiler(self.profile_predict_cfg)
                        self.profile_started_flag = True
                    with record_function("dataloader_iteration"):
                        batch_outputs = self._call_predict(batch_X.cpu())
                    self.profiler.step()
                    self._check_stop_profiler(self.profile_predict_cfg)
                else:
                    batch_outputs = self._call_predict(batch_X.cpu())

                outputs_list.append(batch_outputs)
            outputs = np.concatenate(outputs_list, axis=0)
            y = torch.from_numpy(outputs)
        else:
            y = self._call_predict(X.cpu())

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
        """
        Serialize the wrapped model to a file.

        Parameters
        ----------
        task_id : str
            Identifier used to derive the output path.

        Returns
        -------
        object
            Serialized model reference or path.
        """
        pass

    @abstractmethod
    def load_model(self, task_id):
        """
        Load the wrapped model from a file.

        Parameters
        ----------
        task_id : str
            Identifier used to derive the input path.

        Returns
        -------
        object
            Restored model object.
        """
        pass

    @property
    @abstractmethod
    def allows_multi_target(self) -> bool:
        """Return whether the wrapper supports multi-target prediction."""
        pass

    def _setup_profiler(self, cfg: OmegaConf):
        """
        Create and start a torch profiler from a Hydra-style config.

        Parameters
        ----------
        cfg : OmegaConf
            Profiler schedule and export configuration.
        """
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
        """
        Stop and export the profiler once the configured schedule finishes.

        Parameters
        ----------
        cfg : OmegaConf
            Profiler configuration containing the trace export path.
        """
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
    """
    Base wrapper for feed-forward models used in evaluator-style calls.

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
        """
        Store the backbone and wrapper kwargs for downstream access.

        Parameters
        ----------
        backbone : torch.nn.Module
            Wrapped backbone module.
        **kwargs : dict
            Extra wrapper metadata stored in ``self.kwargs``.
        """
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
        """
        Call ``forward`` and return its model output.

        Parameters
        ----------
        batch : dict
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
        """
        Run the forward pass for one batch.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        """


class AbstractFeedForwardTrainingWrapper(ABC, torch.nn.Module):
    """
    Base wrapper for training-oriented feed-forward model calls.

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
        """
        Store the backbone and wrapper kwargs for training-time use.

        Parameters
        ----------
        backbone : torch.nn.Module
            Wrapped backbone module.
        **kwargs : dict
            Extra wrapper metadata stored in ``self.kwargs``.
        """
        super().__init__()
        self.backbone = backbone
        self.kwargs = AttributeDict(kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(backbone={self.backbone})"

    def __call__(self, batch):
        """
        Call ``forward`` and return its model output.

        Parameters
        ----------
        batch : dict
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
        """
        Run the forward pass for one batch.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.
        """


__all__ = [
    "AbstractFitPredictWrapper",
    "AbstractFeedForwardWrapper",
    "AbstractFeedForwardTrainingWrapper",
]
