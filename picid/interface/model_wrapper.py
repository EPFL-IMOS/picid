from typing import Callable

import torch
from torch import nn

from picid.model.wrappers.base import AbstractFeedForwardTrainingWrapper


def default_pre_process(batch: dict[str, torch.Tensor],
                        to_extract :  str | list[str],
                        **kwargs) -> list[torch.Tensor]:

    if isinstance(to_extract, str):
        to_extract = [to_extract]

    return [batch[t] for t in to_extract]


class ModelWrapper(nn.Module):
    def __init__(self,
                 model : nn.Module,
                 forward_function:  Callable | None = None,
                 pre_process_function:  Callable | None = None,
                 **kwargs):

        super().__init__()
        self.base_model = model

        if forward_function is None:
            forward_function = lambda x: x

        self._forward_function = forward_function

        if pre_process_function is None:
            pre_process_function = lambda x: x

        self._pre_process_function = pre_process_function

    def forward(self, x, **kwargs):
        x = self._pre_process_function(x)
        pred = self.base_model(x)
        pred = self._forward_function(pred)

        return pred


class CustomModelWrapper(AbstractFeedForwardTrainingWrapper):
    def __init__(self,
                 task_type: str,
                 model : nn.Module,
                 pre_process_function: Callable | None = default_pre_process,
                 # post_process_function: Callable | None = None,
                 # model_wrapper: nn.Module = ModelWrapper,
                 **kwargs):

        self.task_type = task_type
        self._pre_process_function = pre_process_function

        # backbone = model_wrapper(model, forward_function=post_process_function,
        #                          parent_model=self)

        super().__init__(model, **kwargs)

    def forward(self, batch: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        x, y = self._pre_process_function(batch, to_extract=['features', self.task_type], **kwargs)

        outputs = self.backbone(x, batch=batch, **kwargs)

        model_out = {
            "predictions": outputs,
            "targets": y,
        }

        return model_out
