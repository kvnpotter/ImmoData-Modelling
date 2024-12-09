# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import shap

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

    def correlation_heatmap(
        self,
        columns: list[str],
        title: str,
        method: str = "spearman",
        view: bool = False,
    ) -> None:
        """
        Construct a heatmap of correlations.

        : param columns: list[str]: List of strings of columns to use for calculation and vis.
        : param method: str: Which method to use to calculate correlation coefficients. Default is Spearman, others similar to pandas.corr method options.
        : param title: str: Name to give to the created file
        : param view: bool: Show the created plot or not, default False.
        """
        data_cols = self.data[columns]
        correlations = data_cols.corr(method=method)
        sns.heatmap(correlations, annot=True, cmap="coolwarm")
        plt.title(title)
        plt.savefig(f"./Results-Graphs/{title}.png", dpi=300)
        if view == True:
            plt.show()


class PCA_Visualiser:
    """
    A class to allow construction of data visualisation for useful visualisations when performing PCA.
    """

    def __init__(self, pca: PCA, data: pd.DataFrame) -> None:
        """
        Create an instance of PCA_Visualiser.

        : param data: pd.DataFrame: Dataframe containing data
        """
        self.pca = pca
        self.data = data

    def broken_stick(self, view: bool = False) -> None:
        """
        Generate broken stick model for PCA and component selection

        : param view: bool: Show graph ; default False.
        """

        # Create Broken stick model to visually assess how many components to extract

        def broken_stick(n_components, total_components):
            """Calculate broken-stick model values for n_components."""
            return [
                np.sum([1 / k for k in range(i, total_components + 1)])
                / total_components
                for i in range(1, n_components + 1)
            ]

        n_components = len(self.pca.explained_variance_ratio_)

        if self.data.shape[0] <= self.data.shape[1]:
            total_components = self.data.shape[0] - 1
        else:
            total_components = self.data.shape[1]

        broken_stick_values = broken_stick(n_components, total_components)

        # Step 5: Create a DataFrame for Comparison
        explained_variance = self.pca.explained_variance_ratio_

        # Step 6: Plot Results
        plt.figure(figsize=(8, 5))
        plt.plot(
            range(1, n_components + 1),
            explained_variance,
            "o-",
            label="Observed Variance",
        )
        plt.plot(
            range(1, n_components + 1),
            broken_stick_values,
            "o-",
            label="Broken-Stick Model",
        )
        plt.axhline(y=0, color="black", linestyle="--", linewidth=0.7)
        plt.xlabel("Principal Component")
        plt.ylabel("Proportion of Variance Explained")
        plt.title("Broken Stick Model vs Observed Eigenvalues")
        plt.legend()
        plt.grid()

        plt.savefig(f"./Results-Graphs/Broken_stick_model.png", dpi=300)

        if view == True:
            plt.show()

        # Print components that exceed broken stick model
        print(
            "\
        Components exceeding broken stick model:"
        )
        for i in range(n_components):
            if self.pca.explained_variance_ratio_[i] > broken_stick_values[i]:
                print(
                    f"PC{i+1}: Observed = {self.pca.explained_variance_ratio_[i]:.3f}, Broken Stick = {broken_stick_values[i]:.3f}"
                )

    def loadings(self, loadings: pd.DataFrame, view: bool = False) -> None:
        """
        Construct a heatmap of factor loadings.

        : param loadings: pd.DataFrame: Dataframe of factor loadings.
        : param view: bool: Show created plot, default False.
        """

        plt.figure(figsize=(10, 8))
        sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0)
        plt.title("PCA Component Loadings")

        plt.savefig(f"./Results-Graphs/Loadings.png", dpi=300)

        if view == True:
            plt.show()

    def correlation_components(self, scores: pd.DataFrame, view: bool = False) -> None:
        """
        Construct a heatmap of correlation between components and response variable.

        : param scores: pd.DataFrame: Dataframe containing factor scores of observations and the response variable.
        : param view: bool: Show plot, default False.
        """

        # Calculate Spearman correlation
        correlation_matrix = scores.corr(method="spearman")

        # Plot the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True
        )
        plt.title("Spearman Correlation Heatmap", fontsize=16)
        plt.tight_layout()

        plt.savefig(f"./Results-Graphs/PCA_correlations.png", dpi=300)

        if view == True:
            plt.show()


class SHAP_Visualiser:
    """
    Class to implement visualisations for SHAP values.
    """

    def __init__(self, SHAP_values, X_test: np.ndarray) -> None:
        """
        Use passed SHAP values as argument to create visualisations

        :param SHAP_values: ...: SHAP values from Evaluator
        :param X_test: np.ndarray: Test dataset of predictors.
        """

        shap.summary_plot(SHAP_values, X_test)
