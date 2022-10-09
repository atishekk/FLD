import xgboost as xgb

class GradientBoosting:
    def __init__(self, params, num_rounds, metrics) -> None:
        self.params = params
        self.num_rounds = num_rounds
        self.metrics = metrics

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, y)
        self.model = xgb.train(self.params, dtrain, self.num_rounds, self.metrics)
        return self.model
