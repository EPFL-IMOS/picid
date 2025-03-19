from abc import ABC
from einops import rearrange
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from tqdm import tqdm
import time


import warnings

warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention.")

from lmetk.utils.logging import MetricLogger, SmoothedValue


class GenericTrainingStrategy(ABC):
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
                 batched_sample_inference=None,
                 N_samples=None):
        super().__init__(cfg, objs, lags)

        self.feature_mode = feature_mode
        self.batched_sample_inference = batched_sample_inference
        self.N_samples = N_samples
        self.dm = objs["dm"]
        self.model = objs["model"]
        self.seed = cfg.glob.seed
        self.cfg = cfg
        
        
    def run(self):
        results = []
        predictions = {}
        log_writer = self.objs['log_writer']

        seed = self.seed
        dm = self.dm
        N_samples = self.N_samples
        n_steps = self.cfg.datamodule.ts_out
        print_freq = 1
        var_names = [dm.target_col] if hasattr(dm, "target_col") else None

        for id_step, step in enumerate(range(n_steps)):
            header = 'Lag: [{}]'.format(step)
            metric_logger = MetricLogger(delimiter="  ")
            metric_logger.add_meter('mse', SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('r2', SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('fit_time', SmoothedValue(window_size=1, fmt='{value:.6f}'))

            X_train, y_train = dm.to_sklearn(dl=dm.train_dataloader(), lag=step)
            X_test, y_test = dm.to_sklearn(dl=dm.test_dataloader(), lag=step)
            
            if var_names is None:
                var_names = [f"Var_{i}" for i in range(y_train.shape[1])]

            if N_samples is not None:
                # Shape: (N_samples, n_features)
                X_train, y_train = X_train[-N_samples:], y_train[-N_samples:]
                X_test, y_test = X_test[:N_samples], y_test[:N_samples]

            for id_var, variable in enumerate(metric_logger.log_every(
                    range(y_train.shape[1]), print_freq, header)):
                assert len(X_train) == len(y_train)
                var_name = var_names[id_var]
                y_train_var = rearrange(y_train[:, variable], "B -> B 1")
                y_test_var = rearrange(y_test[:, variable], "B -> B 1")

                fit_stats = self.fit(X_train, rearrange(y_train_var, "N 1 -> N"))
                pred = rearrange(self.predict(X_test), "N -> N 1")
                r = self.evaluate(y_test_var, pred)
                predictions[(var_name, step + 1)] = pred.squeeze()

                metric_logger.update(
                    mse=r['mse'], 
                    r2=r['r2'], 
                    fit_time=fit_stats['fit_time'])
                
                log_writer.update(head=f"var_{var_name}",
                                    step=step + 1,
                                    mape=r['mape'],
                                    mae=r['mae'],
                                    mse=r['mse'],
                                    r2=r['r2'])

                results.append({
                    "seed": seed,
                    "step": step + 1,
                    "variable": var_name,
                    **r,
                })

        df = pd.DataFrame(results)
        df.to_csv(self.cfg.dirs.out_dir + "/results.csv", index=False)

        df_predictions = pd.DataFrame(predictions)
        df_predictions.to_pickle(self.cfg.dirs.out_dir + "/predictions.pkl")

        mean_per_var = df.groupby(by=["variable"]).mean()
        log_writer.update(head="mean_vars",
                          step=n_steps,
                          **{f'var_{i}: {col}': mean_per_var.at[i, col] for i in mean_per_var.index for col in ['mape', 'mae', 'mse', 'r2']})

        mean_overall = mean_per_var.mean()
        log_writer.update(head="mean_overall",
                          step=n_steps,
                          mape=mean_overall['mape'],
                          mae=mean_overall['mae'],
                          mse=mean_overall['mse'],
                          r2=mean_overall['r2'])


    def fit(self, X, y):
        start_time = time.time()
        regressor = self.model
        regressor.fit(X, y)
        end_time = time.time()
        runtime = end_time - start_time
        fit_stats = {
            'fit_time': runtime,
        }
        return fit_stats
        

    def predict(self, X):
        # we test single sample inference since the model tokenizes all y values
        if self.batched_sample_inference is not None:
            results = []
            num_samples = X.shape[0]
            for i in range(0, num_samples, self.batched_sample_inference):
                x_batch = X[i : i + self.batched_sample_inference, :]  # Handles last batch size automatically
                y_batch = self.model.predict(x_batch)
                results.append(y_batch)
            return np.concatenate(results)
            
        else:
            return self.model.predict(X)

    def evaluate(self, y, preds):
        # Evaluate the models
        mape = 100 * mean_absolute_percentage_error(y, preds)  #  factor 100 due to unusual sklearn convention
        mae = mean_absolute_error(y, preds)
        mse = mean_squared_error(y, preds)
        r2 = r2_score(y, preds)

        results = {
            "mape": mape.astype(float),
            "mae": mae.astype(float),
            "mse": mse.astype(float),
            "r2": r2.astype(float),
            **self._custom_metrics(y, preds),
        }

        return results

    def _custom_metrics(self, y, preds):
        return {}
