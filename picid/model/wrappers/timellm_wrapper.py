from lmetk.baselines.definitions import REGRESSION_TASKS, CLASSIFICATION_TASKS, FORECASTING_TASKS
from lmetk.model.methods.cnn1d import CNN
from lmetk.model.wrappers.base import (
    AbstractFeedForwardTrainingWrapper,
    AbstractFeedForwardWrapper,
)


class TimeLLMWrapper(AbstractFeedForwardTrainingWrapper):
    """Wrapper for the TimeLLM model."""

    def __init__(self, task_type: str, seq_len: int, **kwargs):

        # Define supported tasks based on your new categories
        self.regression_tasks = FORECASTING_TASKS
        self.classification_tasks = None
        self.supported_types = self.regression_tasks

        if task_type not in self.supported_types:
            raise ValueError(f"Task {task_type} not supported for TimeLLMWrapper.")

        self.task_type = task_type
        self.seq_len = seq_len

        # TODO: Initialize the backbone model here
        backbone = CNN(
            seq_len=seq_len,
            in_channels=kwargs["in_channels"],
            cls_head=kwargs.get("cls_head", False),
            num_classes=kwargs.get("num_classes", 2),
            latent_dim=kwargs.get("latent_dim", 256),
            cls_weight=kwargs.get("cls_weight", 1.0),
        )
        super().__init__(backbone=backbone, **kwargs)

    def forward(self, batch):
        """
        Perform a forward pass through the model.
        Args:
            batch (dict): A dictionary containing the input data and target values.
                          Expected keys are:
                          - "batch_x" (torch.Tensor): The input tensor of shape
                            (batch_size, sequence_length, input_features).
                          - "batch_y" (torch.Tensor): The target tensor of shape
                            (batch_size, sequence_length, target_features).
        Returns:
            dict: A dictionary containing:
                  - "predictions" (torch.Tensor): The model's predictions of shape
                    (batch_size, pred_len, output_features), where `pred_len` is
                    determined by `self.kwargs.pred_len`.
                  - "targets" (torch.Tensor): The corresponding target values of shape
                    (batch_size, pred_len, target_features).
        """

        if self.task_type in self.supported_types:

            # The model expects input of shape (batch_size, in_channels, seq_len)
            # but the input is of shape (batch_size, seq_len, in_channels)
            # so we need to permute the dimensions

            # batch_x = batch["target"]["target_seq_x"]
            # batch_y = batch["target"]["target_seq_y"]
            # batch_x_mark = batch["context"]["features_seq_x"]
            # batch_y_mark = batch["context"]["features_seq_y"]

            # TODO: call the model to train
            batch_x = batch["features"].permute(0, 2, 1)
            batch_y = batch[self.task_type]
            outputs = self.backbone(batch_x)
            model_out = {
                "predictions": outputs.unsqueeze(1),
                "targets": batch_y,
            }

            # Make sure the shapes of predictions and targets are correct
            if model_out["predictions"].shape != model_out["targets"].shape:
                raise ValueError(
                    f"Shape mismatch: predictions {model_out['predictions'].shape} "
                    f"and targets {model_out['targets'].shape}."
                )
            return model_out
        else:
            raise ValueError(f"Task {self.task_type} not supported.")
