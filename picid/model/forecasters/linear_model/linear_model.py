from typing import Callable

import torch
from picid.model.forecasters.forecaster import Forecaster

from .linear_ar import LinearModel


class Linear_Forecaster(Forecaster):
    def __init__(
        self,
        optimizer_factory: Callable[..., torch.optim.Optimizer],
        scheduler_factory: Callable[..., torch.optim.lr_scheduler._LRScheduler],
        d_x: int,
        d_yc: int,
        d_yt: int,
        context_points: int,
        task_type: str,
        loss: str = "mse",
        linear_window: int = 0,
        linear_shared_weights: bool = False,
        use_revin: bool = False,
        use_seasonal_decomp: bool = False,
        **kwargs,
    ):
        super().__init__(
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            d_x=d_x,
            d_yc=d_yc,
            d_yt=d_yt,
            loss=loss,
            linear_window=linear_window,
            linear_shared_weights=linear_shared_weights,
            use_revin=use_revin,
            use_seasonal_decomp=use_seasonal_decomp,
            evaluators=kwargs.get("evaluators", None),
            task_type=task_type,
        )

        self.model = LinearModel(
            context_points, shared_weights=linear_shared_weights, d_yt=d_yt
        )

    @property
    def eval_step_forward_kwargs(self):
        return {}

    @property
    def train_step_forward_kwargs(self):
        return {}

    def forward_model_pass(self, x_c, y_c, x_t, y_t):
        _, pred_len, d_yt = y_t.shape
        output = self.model(y_c, pred_len=pred_len, d_yt=d_yt)
        return (output,)
