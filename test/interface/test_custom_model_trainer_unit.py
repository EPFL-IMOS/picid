import torch

from picid.interface.model.custom_model import CustomModelTrainer


class _RegressionBackbone(torch.nn.Module):
    def forward(self, x, batch=None, **kwargs):
        return x.mean(dim=(1, 2))


class _ClassificationBackbone(torch.nn.Module):
    def forward(self, x, batch=None, **kwargs):
        batch_size = x.shape[0]
        return torch.arange(batch_size * 3, dtype=x.dtype).reshape(batch_size, 3)


def test_custom_model_trainer_regression_uses_model_definitions_task_groups():
    trainer = CustomModelTrainer(
        task_type="rul",
        model=_RegressionBackbone(),
    )
    batch = {
        "features": torch.randn(4, 6, 2),
        "rul": torch.randn(4),
    }

    model_out = trainer(batch)

    assert model_out["predictions"].shape == (4, 1, 1)
    assert model_out["targets"].shape == (4, 1)


def test_custom_model_trainer_classification_uses_model_definitions_task_groups():
    trainer = CustomModelTrainer(
        task_type="fault_classification",
        model=_ClassificationBackbone(),
    )
    batch = {
        "features": torch.randn(2, 5, 3),
        "fault_classification": torch.tensor([0, 1]),
    }

    model_out = trainer(batch)

    assert model_out["predictions"].shape == (2, 1, 3)
    assert model_out["targets"].shape == (2, 1)
