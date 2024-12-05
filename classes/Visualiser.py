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
        pass

    def correlation_heatmap(self, data: pd.DataFrame, columns: list[str], method= 'spearman': str) -> None:
        """
        Construct a heatmap of correlations.

        : param data: pd.DataFrame: Dataframe (from an instance of another class) used to calculate correlations.
        : param columns: list[str]: List of strings of columns to use for calculation and vis.
        : param method: str: Which method to use to calculate correlation coefficients. Default is Spearman, others similar to pandas.corr method options.
        """
        data_cols = data[columns]
        correlations = data_cols.corr(method= method)
        sns.heatmap(correlations, annot= True, cmap= 'coolwarm')
        plt.title(f"{method} - correlations between numerical variables")
        plt.show()