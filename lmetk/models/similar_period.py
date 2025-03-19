import numpy as np
from sklearn.neighbors import KNeighborsRegressor


class SimilarPeriod:
    def __init__(self, n_neighbors=1):
        self.regressor = KNeighborsRegressor(n_neighbors=n_neighbors)

    def fit(self, X_context, y_context):
        self.regressor.fit(X_context, y_context)

    def predict(self, X):
        return self.regressor.predict(X)