# Imports

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import r2_score

# Classes for model evaluation

class Evaluator:
    """
    Class implementing all methods to evaluate obtained models.
    """

    def __init__(self, model: KNeighborsRegressor, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Instantiate an object of class Evaluator

        : param model: KNeighborsRegressor: Model to evaluate.
        : param X_train: np.ndarray: Training X dataset. Distance matrix if Gower distance used.
        : param y_train: np.ndarray: Training y dataset. 
        : param X_test: np.ndarray: Testing X dataset. Distance matrix if Gower distance used.
        : param y_test: np.ndarray: Testing y dataset.
        """

        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def adjusted_r2_scorer(model: KNeighborsRegressor, X: np.ndarray, y_true: np.ndarray):
        """Custom scorer for adjusted R²."""
        y_pred = estimator.predict(X)
        n = X.shape[0]  # Number of samples
        p = X.shape[1]  # Number of features
        r2 = r2_score(y_true, y_pred)
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    def get_adj_r2(self, training: bool = True):

        