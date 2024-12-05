# Imports

import pandas as pd

# Classes for handling modelling

class Modeller:
    """
    Class for handling all modelling options.
    """
    def __init__(self):
        pass

    def knn_gower_cv(self, data: pd.DataFrame) -> None:
        """
        Perform KNN regression using a precomputed Gower distance. Execute hyperparameter tuning of n_neighbors, weighting 

        Args:
            data (pd.DataFrame): _description_
        """
