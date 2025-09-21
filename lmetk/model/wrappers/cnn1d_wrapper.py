from lmetk.model.methods.cnn1d import CNN
from lmetk.model.wrappers.base import AbstractFeedForwardWrapper


class CNN1D_Wrapper(AbstractFeedForwardWrapper):
    """Wrapper for the CNN1D model."""

    def __init__(self, task_type: str, seq_len: int, **kwargs):

        if task_type not in ["rul"]:
            raise ValueError(f"Task {task_type} not supported for CNN1D_Wrapper.")

        self.task_type = task_type
        self.seq_len = seq_len
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

        if self.task_type == "rul":
            batch_x = batch["features"]
            batch_y = batch["rul"]
            outputs = self.backbone(batch_x)
            model_out = {
                "predictions": outputs.unsqueeze(1),
                "targets": batch_y.unsqueeze(1),
            }
            return model_out
        else:
            raise ValueError(f"Task {self.task_type} not supported.")
