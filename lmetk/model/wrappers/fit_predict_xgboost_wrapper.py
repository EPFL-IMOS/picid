from pathlib import Path
import joblib
from overrides import override
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from lmetk.model.wrappers.base import AbstractFitPredictWrapper


class FitPredictXGBoostWrapper(AbstractFitPredictWrapper):
    """Wrapper for the XGboost model."""

    def __init__(self, model_cache_path: str = None, **kwargs):
        backbone = MultiOutputRegressor(
            GradientBoostingRegressor(random_state=42, verbose=2)
        )

        self.model_cache_path = model_cache_path
        super().__init__(backbone=backbone, **kwargs)

    @override
    def serialize_model(self, task_id):
        """
        Serialize the model to a file using joblib.
        depending on task_id and self.kwargs['model_path'].
        """
        model_path = self.model_cache_path
        if model_path is None and task_id is not None:
            model_path = str(Path(".model_cache") / f"{task_id}.joblib")

        if model_path is None:
            raise ValueError("No model_path provided in kwargs and no task_id given.")

        # Ensure parent directory exists
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        # Use joblib to persist the estimator which may contain torch objects
        joblib.dump(self.backbone, model_path)
        return model_path

    @override
    def load_model(self, task_id):
        """
        Load the model from a file using joblib.
        """
        model_path = self.model_cache_path
        if model_path is None and task_id is not None:
            model_path = str(Path(".model_cache") / f"{task_id}.joblib")

        if model_path is None:
            raise ValueError("No model_path provided in kwargs and no task_id given.")

        return joblib.load(model_path)

    @property
    @override
    def allows_multi_target(self) -> bool:
        """Indicates whether the model supports multi-target prediction."""
        return True
