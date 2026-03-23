from typing import Literal

from pydantic import Field, model_validator, BaseModel


class TrainerConfig(BaseModel):
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
        assert self.min_epochs < self.max_epochs, 'min_epochs must be lower thant max_epochs'

        return self
