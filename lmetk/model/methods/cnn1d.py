# Reference model taken from
# https://github.com/EPFL-IMOS/concept-prognostics/blob/main/cem/models/cnn.py

import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    A 1D Convolutional Neural Network (CNN) model for regression and optional classification tasks.
    Args:
        in_channels (int): Number of input channels. Must be less than 20.
        cls_head (bool, optional): If True, includes a classification head. Defaults to False.
        num_classes (int, optional): Number of classes for the classification head. Defaults to 2.
        latent_dim (int, optional): Dimension of the latent feature space. Defaults to 256.
        cls_weight (float, optional): Weight for the classification loss. Defaults to 1.0.
    Attributes:
        cls_head (bool): Indicates if the classification head is included.
        cls_weight (float): Weight for the classification loss.
        latent_dim (int): Dimension of the latent feature space.
        backbone (torch.nn.Sequential): The feature extraction backbone consisting of convolutional layers.
        regression (torch.nn.Sequential): The regression head for predicting continuous values.
        classification (torch.nn.Sequential, optional): The classification head for predicting class probabilities.
        loss_cls (torch.nn.Module, optional): The loss function for the classification task.
    Methods:
        forward(x):
            Forward pass of the model. Returns regression output and optionally classification output.
    Raises:
        AssertionError: If `in_channels` is not less than 20.
    """

    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        cls_head=False,
        num_classes=2,
        latent_dim=256,
        cls_weight=1.0,
    ):
        super().__init__()
        self.cls_head = cls_head
        self.cls_weight = cls_weight
        self.latent_dim = latent_dim

        assert in_channels < 20, "in_channels should be < 20"

        self.cnn_backbone = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels, out_channels=20, kernel_size=3, padding="same"
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=20, out_channels=20, kernel_size=3, padding="same"
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=20, out_channels=10, kernel_size=3, padding="same"
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=10, out_channels=10, kernel_size=3, padding="same"
            ),
            torch.nn.ReLU(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(
                1, in_channels, seq_len
            )  # Assuming input length is 100
            dummy_output = self.cnn_backbone(dummy_input)
        n_features = dummy_output.shape[1] * dummy_output.shape[2]

        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_features, self.latent_dim),
            torch.nn.ReLU(),
        )
        self.regression = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 1), torch.nn.Flatten(start_dim=0)
        )

        if self.cls_head:
            self.classification = torch.nn.Sequential(
                torch.nn.Linear(self.latent_dim, num_classes)
            )
            self.loss_cls = (
                torch.nn.BCEWithLogitsLoss()
            )  # or CE if mutually exclusive concepts!

    def forward(self, x):
        feat = self.cnn_backbone(x)
        feat = self.head(feat)
        regres = self.regression(feat)
        if self.cls_head:
            classi = self.classification(feat)
            # return regres, feat, classi
            return regres, classi
        # return regres, feat
        return regres
