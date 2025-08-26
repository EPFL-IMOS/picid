"""Wrapper for the TimePFN model."""
from typing import Optional, Any

import torch
from lmetk.model.wrappers.base import AbstractWrapper

from omegaconf import OmegaConf
from autogluon.tabular import TabularPredictor
from autogluon.features.generators import AsTypeFeatureGenerator, IdentityFeatureGenerator, BulkFeatureGenerator

import numpy as np
import pandas as pd

from einops import rearrange

from lmetk.model.autogluon.multi_label_predictor import MultilabelTabularPredictor


class AutoGluonWrapper(AbstractWrapper):
    r"""Wrapper for the TimePFN model.

    This wrapper defines the forward pass of the TimePFN model.
    """
    def __init__(self, presets: str = "best",
                 problem_type: str = "regression",
                 holdout_frac: Optional[float] = None,
                 num_bag_folds: int = 0,
                 num_stack_levels: int = 0,
                 hyperparameter_tune_kwargs: dict[str, float] = None,
                 model_dict: dict[str, dict] = None,
                 generator_list: Optional[Any] = None,
                 save_space: bool = False,
                 output_dir: str = "testwdir/autogluon",
                 **kwargs):

        self.n_targets = kwargs['train_dataloader'].dataset.y.shape[2]
        self.label_names = [f"target_{i}" for i in range(self.n_targets)]
        self.problem_type = problem_type
        self.output_dir = output_dir

        backbone = self._initialize_backbone()
        super().__init__(backbone=backbone, **kwargs)
        self.model_fit_flag = False
        self.lag = 0

        self.presets = presets

        self.holdout_frac = holdout_frac  # validation set size
        self.num_bag_folds = num_bag_folds  # cross-validation ensembling
        self.num_stack_levels = num_stack_levels  # stacking layers
        self.hyperparameter_tune_kwargs = hyperparameter_tune_kwargs

        self.generator_list = [AsTypeFeatureGenerator(), IdentityFeatureGenerator()] if generator_list is None else generator_list
        self.feature_generators = self._initialize_feature_generators()

        self.model_dict = OmegaConf.to_container(model_dict, resolve=True)
        self.save_space = save_space

    def _initialize_backbone(self):
        if self.n_targets > 1:
            return MultilabelTabularPredictor(labels=self.label_names,
                                              problem_type=self.problem_type,
                                              path=self.output_dir)
        else:
            return TabularPredictor(label=self.label_names[0],
                                    problem_type=self.problem_type,
                                    path=self.output_dir)

    def _initialize_feature_generators(self):
        return BulkFeatureGenerator(generators=[type(generator)() for generator in self.generator_list])

    def fit(self, x, y):
        if self.backbone.is_fit:
                self.backbone = self._initialize_backbone()
                self.feature_generators = self._initialize_feature_generators()

        df = pd.DataFrame(x)
        df[self.label_names] = y
        self.backbone.fit(df,
                          presets=self.presets,
                          holdout_frac=self.holdout_frac,
                          num_bag_folds=self.num_bag_folds,
                          num_stack_levels=self.num_stack_levels,
                          hyperparameter_tune_kwargs=self.hyperparameter_tune_kwargs,
                          hyperparameters=self.model_dict,
                          feature_generator=self.feature_generators,
                          save_space=self.save_space)

    def predict(self, x):
        df = pd.DataFrame(x)
        return self.backbone.predict(df).to_numpy()

    def forward(self, batch):
        r"""Forward pass for the Tune wrapper. Note the following:
        future_context_timestamps := batch['context']['y_mark'] is not used due to possible noise,
        historical_target_timestamps := batch['target']['x_mark']  is redundant with historical_context_timestamps,
        future_target_timestamps := batch['target']['y_mark']  is redundant with future_context_timestamps.

        Parameters
        ----------self.backbone.fit(batch_x, batch_y)
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """

        forward_pattern = 'samples sequence features -> samples (features sequence)'
        inverse_pattern = 'samples (features sequence) -> samples sequence features'

        historical_context_timestamps = rearrange(batch['context']['x_mark'], forward_pattern)

        historical_context = rearrange(batch['context']['x'], forward_pattern)
        future_context = rearrange(batch['context']['y'], forward_pattern)

        historical_target = rearrange(batch['target']['x'], forward_pattern)
        future_target = rearrange(batch['target']['y'], forward_pattern)

        feature_list = [historical_context_timestamps, historical_context, historical_target, future_context]
        X_fit = np.nan_to_num(torch.concatenate(feature_list, axis=1).cpu().numpy())
        y_fit = future_target.cpu().numpy()

        # Fit model
        self.fit(X_fit, y_fit)

        # Predict
        outputs = self.predict(X_fit)

        # Update evaluator with current batch
        n_sequence, n_features = batch['target']['y'].shape[1], batch['target']['y'].shape[2]
        model_out = {
            'predictions': rearrange(torch.Tensor(outputs), inverse_pattern, sequence=n_sequence, features=n_features),
            'targets': batch['target']['y']
        }

        
        return model_out