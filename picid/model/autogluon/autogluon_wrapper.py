"""AutoGluon wrapper for multi-target tabular prediction."""

from typing import Optional, Any


from picid.model.definitions import CLASSIFICATION_TASKS
from picid.model.wrappers.base import AbstractFeedForwardWrapper

from omegaconf import OmegaConf
from autogluon.tabular import TabularPredictor
from autogluon.features.generators import (
    AsTypeFeatureGenerator,
    IdentityFeatureGenerator,
    BulkFeatureGenerator,
)

import pandas as pd


from picid.model.autogluon.multi_label_predictor import MultilabelTabularPredictor


class AutoGluonWrapper(AbstractFeedForwardWrapper):
    """
    Wrap AutoGluon tabular predictors for the project model API.

    The wrapper builds either a single ``TabularPredictor`` or a
    ``MultilabelTabularPredictor`` depending on the number of targets.

    Parameters
    ----------
    presets : str, default="best"
        AutoGluon preset bundle to use.
    task_type : str, default="regression"
        Project task identifier.
    holdout_frac : float, optional
        Validation split fraction passed to AutoGluon.
    num_bag_folds : int, default=0
        Number of bagging folds for AutoGluon.
    num_stack_levels : int, default=0
        Number of stacking levels for AutoGluon.
    hyperparameter_tune_kwargs : dict[str, float], optional
        Hyperparameter tuning configuration.
    model_dict : dict[str, dict], optional
        AutoGluon hyperparameters container.
    generator_list : list[Any], optional
        Feature generator classes to instantiate.
    save_space : bool, default=False
        Whether to ask AutoGluon to free intermediate artifacts.
    output_dir : str, default="testwdir/autogluon"
        Output directory for the AutoGluon run.
    **kwargs : Any
        Additional wrapper arguments.
    """

    def __init__(
        self,
        presets: str = "best",
        task_type: str = "regression",
        holdout_frac: Optional[float] = None,
        num_bag_folds: int = 0,
        num_stack_levels: int = 0,
        hyperparameter_tune_kwargs: dict[str, float] = None,
        model_dict: dict[str, dict] = None,
        generator_list: Optional[Any] = None,
        save_space: bool = False,
        output_dir: str = "testwdir/autogluon",
        **kwargs,
    ):
        # dict_keys(['batch_idx', 'target', 'context'])
        batch = next(iter(kwargs["train_dataloader"]))
        # dict_keys(['features_seq_x', 'features_seq_y'])
        _context = batch["context"]
        # dict_keys(['target_seq_x', 'target_seq_y'])
        target = batch["target"]

        self.n_targets = target["target_seq_x"].shape[2]
        self.label_names = [f"target_{i}" for i in range(self.n_targets)]
        self.task_type = task_type
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

        self.generator_list = (
            [AsTypeFeatureGenerator(), IdentityFeatureGenerator()]
            if generator_list is None
            else generator_list
        )
        self.feature_generators = self._initialize_feature_generators()

        self.model_dict = OmegaConf.to_container(model_dict, resolve=True)
        self.save_space = save_space

    def _initialize_backbone(self):
        if self.task_type in CLASSIFICATION_TASKS:
            problem_type = "classification"
        else:
            problem_type = "binary"

        if self.n_targets > 1:
            return MultilabelTabularPredictor(
                labels=self.label_names,
                problem_type=problem_type,
                path=self.output_dir,
            )
        else:
            return TabularPredictor(
                label=self.label_names[0],
                problem_type=problem_type,
                path=self.output_dir,
            )

    def _initialize_feature_generators(self):
        return BulkFeatureGenerator(
            generators=[type(generator)() for generator in self.generator_list]
        )

    def fit(self, x, y):
        if self.backbone.is_fit:
            self.backbone = self._initialize_backbone()
            self.feature_generators = self._initialize_feature_generators()

        df = pd.DataFrame(x)
        df[self.label_names] = y
        self.backbone.fit(
            df,
            presets=self.presets,
            holdout_frac=self.holdout_frac,
            num_bag_folds=self.num_bag_folds,
            num_stack_levels=self.num_stack_levels,
            hyperparameter_tune_kwargs=self.hyperparameter_tune_kwargs,
            hyperparameters=self.model_dict,
            feature_generator=self.feature_generators,
            save_space=self.save_space,
        )
