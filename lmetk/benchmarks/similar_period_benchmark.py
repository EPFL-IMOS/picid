import warnings

import pandas as pd
import torch
from einops import rearrange
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error

from lmetk.benchmarks.forecasting_benchmark import ForecastingBenchmark

warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention.")

from lmetk.utils.logging import MetricLogger, SmoothedValue

class SimilarPeriodStrategy(ForecastingBenchmark):
    def __init__(self, cfg, objs, contiguous=True, N_samples=None, verbose=False):
        super().__init__(cfg, objs, verbose)
        self.cfg = cfg
        self.objs = objs
        self.N_samples = N_samples
        self.contiguous = contiguous
        self.verbose = verbose

    def setup(self):
        pass

    def run(self):
        results = []
        predictions = {}
        log_writer = self.objs['log_writer']
        self.model = self.objs["model"]


        dm = self.objs["dm"]
        N_samples = self.N_samples
        n_lags = self.cfg.datamodule.ts_in
        n_steps = self.cfg.datamodule.ts_out
        print_freq = 1
        var_names = dm.target_cols if hasattr(dm, "target_cols") else None

        _, _, X_train, y_train = self.slice_dataloader(dl=dm.train_dataloader())
        _, _, X_test, y_test = self.slice_dataloader(dl=dm.test_dataloader())

        X_train = rearrange(X_train, "batch steps features -> batch (features steps)")
        X_test = rearrange(X_test, "batch steps features -> batch (features steps)")

        if N_samples is not None:
            X_train = X_train[:N_samples]
            y_train = y_train[:N_samples]
            X_test = X_test[:N_samples]
            y_test = y_test[:N_samples]

        # for id_var, variable in enumerate(metric_logger.log_every(
        #         range(y_test.shape[1]), print_freq, header)):
        for id_var, variable in enumerate(range(y_test.shape[2])):
            # header = 'Var: [{}]'.format(step)
            metric_logger = MetricLogger(delimiter="  ")
            metric_logger.add_meter('mse', SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('r2', SmoothedValue(window_size=1, fmt='{value:.6f}'))

            var_name = var_names[id_var]

            self.fit(X_train, y_train[:, :,  variable])
            period_pred = self.predict(X_test)

            y_test_var = y_test[:, :, variable]

            for id_step, step in enumerate(range(n_steps)):
                y_test_var_step = rearrange(y_test_var[:, step], "B -> B 1")
                pred = rearrange(period_pred[:, step], "B -> B 1")

                r = self.evaluate(y_test_var_step, pred)
                predictions[(var_name, step + 1)] = pred.squeeze()

                metric_logger.update(mse=r['mse'], r2=r['r2'])
                log_writer.update(head=f"Var: {var_name}",
                                  step=step + 1,
                                  mape=r['mape'],
                                  mae=r['mae'],
                                  mse=r['mse'],
                                  r2=r['r2'])
                results.append({
                    "step": step + 1,
                    "variable": var_name,
                    **r})

        df = pd.DataFrame(results)
        df.to_csv(self.cfg.dirs.out_dir + "/results.csv", index=False)

        df_predictions = pd.DataFrame(predictions)
        df_predictions.to_pickle(self.cfg.dirs.out_dir + "/predictions.pkl")

        mean_per_var = df.groupby(by=["variable"]).mean()
        log_writer.update(head="Mean Vars",
                          step=n_steps,
                          **{f'Var {i}: {col}': mean_per_var.at[i, col] for i in mean_per_var.index for col in
                             ['mape', 'mae', 'mse', 'r2']})

        mean_overall = mean_per_var.mean()
        log_writer.update(head="Mean Overall",
                          step=n_steps,
                          mape=mean_overall['mape'],
                          mae=mean_overall['mae'],
                          mse=mean_overall['mse'],
                          r2=mean_overall['r2'])

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

    def save(self):
        pass

    def load(self):
        pass

    def __str__(self):
        return f"Experiment: {self.cfg.info.exp_name}"

    def slice_dataloader(self, dl):
        X_context, y_context, X_target, y_target = [], [], [], []
        for batch in iter(dl):
            x_c, y_c, x_t, y_t = batch
            X_context.append(x_c)
            y_context.append(y_c)
            X_target.append(x_t)
            y_target.append(y_t)

        return torch.concat(X_context).numpy(), torch.concat(y_context).numpy(), torch.concat(X_target).numpy(), torch.concat(y_target).numpy()

    def fit(self, X_context, y_context):
        self.model.fit(X_context, y_context)

    def predict(self, X):
        return self.model.predict(X)