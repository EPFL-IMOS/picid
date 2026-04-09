from typing import Literal

from pydantic import Field, model_validator, BaseModel


class TrainerConfig(BaseModel):
    """Configuration for the PyTorch Lightning Trainer.

    Pass an instance as ``training_config=`` to ``EntryInterface.train()``.
    When omitted, the project's default trainer YAML is used instead.
    For one-off tweaks you can also use raw Hydra override strings (e.g.
    ``overrides=["trainer.max_epochs=50", "trainer.accelerator=gpu"]``).

    Parameters
    ----------
    max_epochs : int
        Maximum number of training epochs. Must be greater than ``min_epochs``.
        Default ``10``.
    min_epochs : int
        Minimum number of training epochs. Default ``1``.
    accelerator : {"cpu", "gpu"}
        Hardware accelerator to use. Default ``"cpu"``.
    devices : list[int]
        Device indices. For GPU training, e.g. ``[0]`` selects the first GPU.
        Default ``[0]``.
    check_val_every_n_epoch : int
        Run validation every N epochs. ``0`` means after every epoch.
        Default ``0``.
    deterministic : bool
        Enable deterministic algorithms for reproducibility. Default ``True``.
    inference_mode : bool
        Use ``torch.inference_mode()`` during evaluation passes. Default ``True``.

    Examples
    --------
    >>> cfg = TrainerConfig(max_epochs=50, accelerator="gpu", devices=[0])
    >>> interface.train(..., training_config=cfg)
    """

    model_class: str = Field('lightning.pytorch.trainer.Trainer',
                                frozen=True,
                             serialization_alias='_target_')

    default_root_dir: str = '${paths.output_dir}'

    min_epochs: int = Field(1, ge=1)
    max_epochs: int = Field(10, ge=1)

    accelerator: Literal['cpu', 'gpu'] = 'cpu'
    devices: list[int] = [0]

    check_val_every_n_epoch: int = Field(0, ge=0)
    deterministic: bool = True
    inference_mode: bool = True

    @model_validator(mode='after')
    def propagate_tag(self) -> 'TrainerConfig':
        assert self.min_epochs < self.max_epochs, 'min_epochs must be lower than max_epochs'

        return self
