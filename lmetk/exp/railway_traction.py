from einops import rearrange
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tqdm import tqdm

# Assuming there is a TabPFNRegressor (if not, a different regressor should be used)
from tabpfn import TabPFNRegressor

from lmetk.case_studies.railway_traction import SpatialDataModule
from lmetk.exp.generic_experiment import GenericExperiment  

import pandas as pd
import torch


def dl_to_sklearn__with_context_for_tabpfn(dl, lag=0):
    X, y = [], []
    for batch in iter(dl): 
        x_c, y_c, x_t, y_t = batch
        X.append(torch.concat([x_t[:, lag, :], y_c.squeeze(2)], dim=1))
        y.append(y_t[:,lag,:])
        
    
    return torch.concat(X).numpy(), torch.concat(y).numpy()


class RailwayTractionExperiment(GenericExperiment):
    
    
    def __init__(self, cfg, objs):
        super().__init__(cfg, objs)
        self.feature_mode = "exogen"  # "endogen" or "exogen"
        self.N_samples = 10
        self.dm = objs["dm"]
        self.model = objs["model"]
        
    
    def setup(self, stage=None):
        self.dm.setup()
        

    def run(self):
        feature_mode = "exogen" 
        model_name = "tabpfn"
        seed = -1
        results = []
        
        # factor_agg = pd.read_csv("datasets/railway_agg.csv")['Bahnlast'].abs().mean()
        factor_full = pd.read_csv("datasets/railway.csv")['Bahnlast'].abs().mean()
        
        for lag in tqdm(range(24)):
            
            N_samples = self.N_samples
            dm = self.dm
            
            # if feature_mode == "endogen":
                # X_train, y_train = dl_to_sklearn_no_context(dm.train_dataloader())
                # X_test, y_test = dl_to_sklearn_no_context(dm.test_dataloader())
            if self.feature_mode == "exogen":
                X_train, y_train = dl_to_sklearn__with_context_for_tabpfn(dm.train_dataloader(), lag=lag)
                X_test, y_test = dl_to_sklearn__with_context_for_tabpfn(dm.test_dataloader(), lag=lag)
            else:
                raise ValueError("Feature mode unknown")
            
            if N_samples is not None:
                X_train, y_train = X_train[-N_samples:], y_train[-N_samples:]
                X_test, y_test = X_test[:N_samples], y_test[:N_samples]
            
            # Initialize the regressor
            regressor = self.model
            regressor.fit(X_train, y_train)

            # Predict on the test set
            y_test_hat = regressor.predict(X_test)

            # Evaluate the model
            mse = mean_squared_error(y_test, y_test_hat)
            r2 = r2_score(y_test, y_test_hat)

            print("Mean Squared Error (MSE):", mse)
            print("R² Score:", r2)



            y_test_hat_rs = dm.inverse_scale_bahnlast(rearrange(y_test_hat, "B -> (B) 1"))
            y_test_rs = dm.inverse_scale_bahnlast(rearrange(y_test, "B F -> (B F) 1"))



            results.append({"exp_name": f"{model_name}_{feature_mode}",
                            "seed": seed,
                            "lag" : lag,
                            "normalized/test/mae": np.abs((y_test_hat_rs - y_test_rs)).mean() *100 / factor_full, 
                            "normalized/test/mse": np.abs((y_test_hat_rs - y_test_rs)**2).mean() *100 / factor_full, })

        df = pd.DataFrame(results).set_index(["exp_name", "seed"])
        df