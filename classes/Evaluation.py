# Imports

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import r2_score
import shap

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
        self.y_pred_test = self.model.predict(self.X_test)
        self.y_pred_train = self.model.predict(self.X_train)

    def adjusted_r2_scorer(self, test: bool = True) -> float:
        """
        Custom scorer for adjusted R².
        For such a large dataset, n >> p > 1 ; therefore R2 adj ~ R2.

        : param test: bool: If true on test data, otherwise on training set.

        : return: float: Adjusted R2 score.
        """
        if test == True:
            y_true = self.y_test
            y_pred = self.y_pred_test
            n = self.X_test.shape[0]  # Number of samples
            p = self.X_test.shape[1]  # Number of features
        else:
            y_true = self.y_train
            y_pred = self.y_pred_train
            n = self.X_train.shape[0]  # Number of samples
            p = self.X_train.shape[1]  # Number of features
        
        r2 = r2_score(y_true, y_pred)
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    def smape(self, test: bool = True) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error (sMAPE).

        : param test: bool: If true on test data, otherwise on training set.

        :Return : float: sMAPE value.
        """
        if test == True:
            y_true = self.y_test
            y_pred = self.y_pred_test
        else:
            y_true = self.y_train
            y_pred = self.y_pred_train

        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        smape_value = np.mean(numerator / denominator) * 100  # In percentage
        return smape_value

    def model_metrics(self, model_name: str) -> None:
        """
        Get model metrics for evaluation, print on screen and store in an Excel.

        param model_name: str: Name of the current model.
        """

        model_metrics = {}
        model_metrics ['model_name'] = [model_name]
        model_metrics['R2_train'] = [r2_score(self.y_train, self.y_pred_train)]
        model_metrics['MAE_train'] = [MAE(self.y_train, self.y_pred_train)]
        model_metrics['RMSE_train'] = [RMSE(self.y_train, self.y_pred_train)]
        model_metrics['MAPE_train'] = [MAPE(self.y_train, self.y_pred_train)]
        model_metrics['sMAPE_train'] = [self.smape(test= False)]

        model_metrics['R2_test'] = [r2_score(self.y_test, self.y_pred_test)]
        model_metrics['MAE_test'] = [MAE(self.y_test, self.y_pred_test)]
        model_metrics['RMSE_test'] = [RMSE(self.y_test, self.y_pred_test)]
        model_metrics['MAPE_test'] = [MAPE(self.y_test, self.y_pred_test)]
        model_metrics['sMAPE_test'] = [self.smape(test= True)]

        model_metrics_df = pd.DataFrame(model_metrics)

        global_metrics = pd.read_csv('./Results-Graphs/model_metrics.csv')
        global_metrics = pd.concat([global_metrics, model_metrics_df], ignore_index=True)
        global_metrics.to_csv('./Results-Graphs/model_metrics.csv', index= False)

        print(global_metrics)
        print(r2_score(self.y_test, self.y_pred_test))

    def predict_fn(self, distances: np.ndarray):
        """
        Helperfunction for SHAP calculation.

        :param distances: np.ndarray: Array of Gower distances.
        """
        return self.model.predict(distances)

    def shap(self, distances: np.ndarray):
        """
        Get SHAP values for predictors added to model.

        :param distances: np.ndarray: Array of Gower distances.
        """

        explainer = shap.Explainer(self.predict_fn, distances)
        shap_values = explainer.shap_values(self.X_test)
        return shap_values

        