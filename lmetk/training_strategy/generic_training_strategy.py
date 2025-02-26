from einops import rearrange
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
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
        
        seed = self.seed
        dm = self.dm
        N_samples = self.N_samples
        
        
        print_freq = 1
        
        for variable in range(1):
            
            header = 'Variable: [{}]'.format(variable)
            metric_logger = MetricLogger(delimiter="  ")
            metric_logger.add_meter('nmse', SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('nr2', SmoothedValue(window_size=1, fmt='{value:.6f}'))
            
            for id_lag, lag in enumerate(metric_logger.log_every(range(24), print_freq, header)):
                X_train, y_train =  dm.to_sklearn(dl=dm.train_dataloader(), lag=lag)    
                X_test, y_test =  dm.to_sklearn(dl=dm.test_dataloader(), lag=lag)
            
                if N_samples is not None:
                    X_train, y_train = X_train[-N_samples:], y_train[-N_samples:]
                    X_test, y_test = X_test[:N_samples], y_test[:N_samples]
                    
                assert len(X_train) == len(y_train) and y_train.shape[1] == 1    
                
                self.fit(X_train, rearrange(y_train, "N 1 -> N"))
                preds = rearrange(self.predict(X_test), "N -> N 1")
                r = self.evaluate(y_test, preds)
                
                metric_logger.update(nmse=r['nmse'], nr2=r['nr2']) 
                
                
                results.append({
                    "seed": seed,
                    "lag" : lag,
                    "variable": variable,
                    **r,
                })
                
        df = pd.DataFrame(results)
                
    def fit(self, X, y):
        regressor = self.model
        regressor.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, y, preds):
        # Evaluate the model
        mse = mean_squared_error(y, preds)
        r2 = r2_score(y, preds)

        results = {
            "nmse": mse,
            "nr2": r2,
            **self._custom_metrics(y, preds),
        }
        
        return results

    
    def _custom_metrics(y, preds):
        return {}