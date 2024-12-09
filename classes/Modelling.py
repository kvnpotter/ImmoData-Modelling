# Imports

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import r2_score

# Classes for handling modelling

class Modeller:
    """
    Class for handling all modelling options.
    """
    def __init__(self):
        """
        Create an instance of class Modeller. 

        """
        self.model_params = None
        self.gower = None
        self.X_train = None
        self.y_train = None
        self.model = None
        self.best_k = None
        self.best_weight = None
        self.best_dist = None

    def set_parameters(self,CV: bool = True, k : int = 5, distance: str = 'euclidean', weight: str = 'uniform') -> None:
        """
        Choose how to define the model. Verify if KNN regressor carried out with/without CV-Gridsearch.
        With in-built distance metric or precomputed, weighting used, number neuighbors.

        : param CV: bool: If True, carry out CV-gridsearch for hyperparameter tuning.
        : param k: int: Number of neighboUrs.
        : param distance: str: Which distance metric to use.
        : param weight: str: Which weighting to use ; uniform or (inverse) distance.
        """
        if CV == True:
            if distance == 'Gower':
                self.gower = True
                self.model_params = {'n_neighbors': list(range(1,30)),
                            'weights': ['uniform', 'distance']}
            else:
                self.model_params = {'n_neighbors': list(range(1,30)),
                            'metric': ['euclidean', 'cosine'],
                            'weights': ['uniform', 'distance']}
        else:
            if distance == 'Gower':
               self.gower = True
               self.model_params = {'n_neighbors': k,
                          'weights': weight}

            else:
                self.model_params = {'n_neighbors': k,
                          'metric': distance,
                          'weights': weight}

    def get_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Choose model to instantiate based on the entered parameters. Build the model 

        : param X_train: np.ndarray: Array of X training data. Array of Gower distances if modelling to be performed using this distance metric.
        : param y_train: np.ndarray: Array of y training data. 
        """
        self.X_train = X_train
        self.y_train = y_train

        if type(self.model_params['n_neighbors']) == list:
            
            if self.gower == True:
                self.knn_gower_CV(self.X_train, self.y_train)

            else:
                self.knn_CV(self.X_train, self.y_train)

        else:

            if self.gower == True:
                self.knn_gower(self.X_train, self.y_train)

            else:
                self.knn(self.X_train, self.y_train)

    def knn(self, X_data: np.ndarray, y_data: np.ndarray) -> None:
        """
        Build and fit KNN regressor model with parameters specified in object attributes and data as arguments in function call.

        : param X_data: np.ndarray: Array of X data.
        : param y_data: np.ndarray: Array of y data.
        """

        KNNR = KNeighborsRegressor(n_neighbors= self.model_params['n_neighbors'],
                                   metric= self.model_params['metric'],
                                   weights= self.model_params['weights'])
        
        self.model = KNNR.fit(X_data, y_data)

    def knn_gower(self, gow_dist: np.ndarray, y_data: np.ndarray) -> None:
        """
        Build and fit KNN regressor model with parameters specified in object attributes and Gower distance, y_data as arguments in function call.

        : param gow_dist: np.ndarray: Array of X data.
        : param y_data: np.ndarray: Array of y data.
        """

        KNNR_gow = KNeighborsRegressor(n_neighbors= self.model_params['n_neighbors'],
                                       metric= 'precomputed',
                                       weights= self.model_params['weights'])
        
        self.model = KNNR_gow.fit(gow_dist, y_data)

    def knn_CV(self, X_data: np.ndarray, y_data: np.ndarray) -> None:
        """
        Build and fit KNN regressor model with parameters specified in object attributes and data as arguments in function call.
        Use CV-gridsearch to find optimal values for hyperparameters

        : param X_data: np.ndarray: Array of X data.
        : param y_data: np.ndarray: Array of y data.
        """
        KNNR = KNeighborsRegressor()
        grid = GridSearchCV(estimator= KNNR,
                            param_grid= self.model_params,
                            cv= 5,
                            scoring= 'neg_root_mean_squared_error',
                            verbose= 1,
                            n_jobs= -1)
        grid_search = grid.fit(X_data, y_data)

        self.model = grid_search.best_estimator_
        self.best_k = grid_search.best_params_['n_neighbors']
        self.best_weight = grid_search.best_params_['weights']
        self.best_dist = 'precomputed'
        print(grid_search.best_params_)
        print(grid_search.best_score_)

    def knn_gower_CV(self, gow_dist: np.ndarray, y_data: np.ndarray) -> None:
        """
        Build and fit KNN regressor model with parameters specified in object attributes and Gower distance, y_data as arguments in function call.
        Use CV-gridsearch to find optimal values for hyperparameters

        : param gow_dist: np.ndarray: Array of X data.
        : param y_data: np.ndarray: Array of y data.
        """
        KNNR_gow = KNeighborsRegressor(metric= 'precomputed')
        grid = GridSearchCV(estimator= KNNR_gow,
                            param_grid= self.model_params,
                            cv= 5,
                            scoring= 'neg_root_mean_squared_error',
                            verbose= 2,
                            n_jobs= -1)
        grid_search_gower = grid.fit(gow_dist, y_data)

        self.model = grid_search_gower.best_estimator_
        self.best_k = grid_search_gower.best_params_['n_neighbors']
        self.best_weight = grid_search_gower.best_params_['weights']
        self.best_dist = 'precomputed'
        print(grid_search_gower.best_params_)
        print(grid_search_gower.best_score_)







