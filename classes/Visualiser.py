# Imports

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Classes

class EDA_Visualiser:
    """
    A class to allow construction of data visualisation for some EDA.
    """
    def __init__(self, data: pd.DataFrame) -> None:
        """
        Construct an instance of EDA_Visualiser
        """
        self.data = data

    def correlation_heatmap(self, columns: list[str], title: str, method: str = 'spearman') -> None:
        """
        Construct a heatmap of correlations.

        : param columns: list[str]: List of strings of columns to use for calculation and vis.
        : param method: str: Which method to use to calculate correlation coefficients. Default is Spearman, others similar to pandas.corr method options.
        : param title: str: Name to give to the created file"""
        data_cols = self.data[columns]
        correlations = data_cols.corr(method= method)
        sns.heatmap(correlations, annot= True, cmap= 'coolwarm')
        plt.title(f"{method} - correlations between numerical variables")
        plt.savefig(f"./Results-Graphs/{title}.png", dpi=300)
        plt.show()

class PCA_Visualiser:
    """
    A class to allow construction of data visualisation for useful visualisations when performing PCA.
    """
    
    def __init__(self):
        pass