class Persistence:
    def __init__(self, lag_index=-1):
        self.lag_index = lag_index

    def fit(self, X_context, y_context):
        self.pred = y_context[:, self.lag_index, :]

    def predict(self):
        return self.pred