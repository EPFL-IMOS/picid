import numpy as np
import pandas as pd
from einops import rearrange

from lmetk.training_strategy.generic_training_strategy import MultiModelUnivariateForecastingStrategy


# Assuming there is a TabPFNRegressor (if not, a different regressor should be used)


class SolarGenerationStrategy(MultiModelUnivariateForecastingStrategy):
    def __init__(self,
                 cfg,
                 objs,
                 lags,
                 feature_mode="exogen",
                 N_samples=None):
        super().__init__(cfg, objs, lags=lags, feature_mode=feature_mode, N_samples=N_samples)
        self.factor_full = pd.read_csv("datasets/solar_generation.csv")['pv_generation'].abs().mean()

    def _custom_metrics(self, y, preds):
        dm = self.dm
        factor_full = self.factor_full

        y_test_hat_rs = dm.inverse_scale_bahnlast(rearrange(preds, "B 1 -> B 1"))
        y_test_rs = dm.inverse_scale_bahnlast(rearrange(y, "B 1 -> B 1"))

        return {
            "normalized/test/mae": (np.abs((y_test_hat_rs - y_test_rs)).mean() * 100 / factor_full).astype(float),
            "normalized/test/mse": (np.abs((y_test_hat_rs - y_test_rs) ** 2).mean() * 100 / factor_full).astype(float)
        }