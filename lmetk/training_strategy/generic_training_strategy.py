from einops import rearrange
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention.")

from lmetk.utils.logging import MetricLogger, SmoothedValue


class GenericTrainingStrategy():
    def __init__(self, cfg, objs, lags):
        self.cfg = cfg
        self.objs = objs

    def setup(self):
        pass

    def run(self):
        pass

    def evaluate(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def __str__(self):
        return f"Experiment: {self.cfg.info.exp_name}"


class MultiModelUnivariateForecastingStrategy(GenericTrainingStrategy):
    def __init__(self,
                 cfg,
                 objs,
                 lags,
                 feature_mode="exogen",
                 N_samples=None):
        super().__init__(cfg, objs, lags)

        self.feature_mode = feature_mode
        self.N_samples = N_samples
        self.dm = objs["dm"]
        self.model = objs["model"]
        self.seed = cfg.glob.seed
        self.cfg = cfg

    def run(self):
        results = []
        log_writer = self.objs['log_writer']

        seed = self.seed
        dm = self.dm
        N_samples = self.N_samples
        n_lags = self.cfg.datamodule.ts_in
        print_freq = 1

        for id_lag, lag in enumerate(range(n_lags)):
            header = 'Lag: [{}]'.format(lag)
            metric_logger = MetricLogger(delimiter="  ")
            metric_logger.add_meter('nmse', SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('nr2', SmoothedValue(window_size=1, fmt='{value:.6f}'))

            X_train, y_train = dm.to_sklearn(dl=dm.train_dataloader(), lag=lag)
            X_test, y_test = dm.to_sklearn(dl=dm.test_dataloader(), lag=lag)

            if N_samples is not None:
                # Shape: (N_samples, n_features)
                X_train, y_train = X_train[-N_samples:], y_train[-N_samples:]
                X_test, y_test = X_test[:N_samples], y_test[:N_samples]

                for id_var, variable in enumerate(metric_logger.log_every(
                        range(y_train.shape[1]), print_freq, header)):
                    assert len(X_train) == len(y_train)
                    y_train_var = rearrange(y_train[:, variable], "B -> B 1")
                    y_test_var = rearrange(y_test[:, variable], "B -> B 1")

                    self.fit(X_train, rearrange(y_train_var, "N 1 -> N"))
                    preds = rearrange(self.predict(X_test), "N -> N 1")
                    r = self.evaluate(y_test_var, preds)

                    metric_logger.update(nmse=r['nmse'], nr2=r['nr2'])
                    log_writer.update(head=f"Var: [{variable}]",
                                      step=lag + 1,
                                      nmape=r['nmape'],
                                      nmae=r['nmae'],
                                      nmse=r['nmse'],
                                      nr2=r['nr2'])

                    results.append({
                        "seed": seed,
                        "lag": lag,
                        "variable": variable,
                        **r,
                    })

        # df = pd.DataFrame(results)

    def fit(self, X, y):
        regressor = self.model
        regressor.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, y, preds):
        # Evaluate the model
        mape = 100 * mean_absolute_percentage_error(y, preds)  #  factor 100 due to unusual sklearn convention
        mae = mean_absolute_error(y, preds)
        mse = mean_squared_error(y, preds)
        r2 = r2_score(y, preds)

        results = {
            "nmape": mape.astype(float),
            "nmae": mae.astype(float),
            "nmse": mse.astype(float),
            "nr2": r2.astype(float),
            **self._custom_metrics(y, preds),
        }

        return results

    def _custom_metrics(self, y, preds):
        return {}
