# Imports

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from gower import gower_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor

from classes.Visualiser import EDA_Visualiser, PCA_Visualiser

# Class for handling data preprocessing


class DataPreprocessor:
    """
    A class to import data and perform various data preprocessing steps
    """

    def __init__(self) -> None:
        """
        Create a Preprocessor object. Initalise attributes.
        """
        self.data = None
        self.data_income = None
        self.modelling_data_X = None
        self.modelling_data_y = None
        self.gowermat = None
        self.X_train = None
        self.X_train_scaled = None
        self.X_test = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None

    def load_data(self, data_path: str) -> None:
        """
        Load dataset, assign correct types and store in data attribute.

        : param data_path: str: String containing data path.
        """
        data = pd.read_csv(data_path)

        data.drop(columns=["Unnamed: 0", "id"], inplace=True)
        data["Locality"] = data["Locality"].astype("str")
        data["Fully equipped kitchen"] = data["Fully equipped kitchen"].astype("str")
        data["Fireplace"] = data["Fireplace"].astype("str")
        data["Terrace"] = data["Terrace"].astype("str")
        data["Garden"] = data["Garden"].astype("str")
        data["Swimming pool"] = data["Swimming pool"].astype("str")
        data["Furnished"] = data["Furnished"].astype("str")
        data["Number of rooms"] = data["Number of rooms"].astype("int64")
        data["Number of facades"] = data["Number of facades"].astype("int64")

        self.data = data
        print(f"Dataset loaded!\nThe loaded (original) dataset characteristics:\n")
        print(self.data.info())

    def add_mean_income(self) -> None:
        """
        Load datasets to join tables, and add mean income per locality.
        """
        # Import and prepare dataset to join Code INS to Postal code
        data_insee = pd.read_csv("./Data/INSEE_PostCode.csv", encoding="latin-1")
        subset_columns = data_insee.columns[6:]
        data_insee["PostalCodes"] = data_insee[subset_columns].apply(
            lambda row: row.dropna().tolist(), axis=1
        )
        data_insee.drop(columns=data_insee.columns[6:22], inplace=True)

        # Import and prepare dataset from SPF Finances to join on Code INS
        data_fin = pd.read_csv("./Data/SPF_FIN_Stat.csv", encoding="latin-1")

        # Merge two datasets on Code INS, keeping only records that are present in both tables
        # Rows without postal code, or without financial data are not interesting for final join with data
        data_fin_postcode = pd.merge(data_fin, data_insee, how="inner", on="Code INS")

        # Unpack/explode lists of post codes to obtain dataset with one row of info per postal code
        data_fin_postcode_exploded = data_fin_postcode.explode("PostalCodes")

        # Convert post codes to str type and join data to original dataset
        data_fin_postcode_exploded["PostalCodes"] = (
            data_fin_postcode_exploded["PostalCodes"].astype("int").astype("str")
        )
        data_postcodes = pd.merge(
            data_fin_postcode_exploded,
            self.data,
            how="inner",
            left_on="PostalCodes",
            right_on="Locality",
        )

        # Drop duplicate columns, rename columns
        data_postcodes.drop(columns="Entités administratives_x", inplace=True)
        data_postcodes.drop(columns="Locality", inplace=True)
        data_postcodes.drop(columns="Code INS", inplace=True)
        data_postcodes.rename(
            columns={"Entités administratives_y": "Locality"}, inplace=True
        )
        data_postcodes.rename(
            columns={"Nombre d'habitants": "N_Inhabitants"}, inplace=True
        )
        data_postcodes.rename(
            columns={"Revenu total net imposable": "Tot_taxable_income"}, inplace=True
        )
        data_postcodes.rename(
            columns={"Revenu moyen par déclaration": "Mean_income_taxunit"},
            inplace=True,
        )
        data_postcodes.rename(
            columns={"Revenu médian par déclaration": "Median_income_taxunit"},
            inplace=True,
        )
        data_postcodes.rename(
            columns={"Revenu moyen par habitant": "Mean_income_inhabitant"},
            inplace=True,
        )
        data_postcodes.rename(
            columns={"Indice de richesse": "Wealth_index"}, inplace=True
        )
        data_postcodes.rename(columns={"Impôt d'Etat": "State_tax"}, inplace=True)
        data_postcodes.rename(
            columns={"Taxes communales et d'agglomération": "Local_tax"}, inplace=True
        )
        data_postcodes.rename(columns={"Impôt total": "Tot_tax"}, inplace=True)
        data_postcodes.rename(columns={"Langue": "Language"}, inplace=True)
        data_postcodes.rename(columns={"Région": "Region"}, inplace=True)
        data_postcodes.rename(columns={"Arrondissement": "District"}, inplace=True)

        # clean data types
        columns_to_convert = [
            "N_Inhabitants",
            "Tot_taxable_income",
            "State_tax",
            "Local_tax",
            "Tot_tax",
        ]
        data_postcodes[columns_to_convert] = (
            data_postcodes[columns_to_convert]
            .apply(lambda col: col.str.replace(".", "", regex=False))
            .astype(float)
        )
        data_postcodes["N_Inhabitants"] = data_postcodes["N_Inhabitants"].astype(int)
        data_postcodes["Wealth_index"] = data_postcodes["Wealth_index"].astype(float)

        self.data_income = data_postcodes
        print(
            f"Data from SPF Finances added to dataset!\nCharacteristics of complete dataset:\n"
        )
        print(self.data_income.info())

    def view_correlations(
        self, data: None | str = None, columns: None | list = None
    ) -> None:
        """
        Allow visualisation of correlations between variables using heatmap.
        If data argument is None (default), use original dataset, otherwise use specified dataset.
        If columns argument is None (default), use all columns.

        : param data: NoneType or str: Specifies which dataset to use. None is default and uses original dataset. Other options are data_income,
        : param columns: NoneType or list: List of columns to use for creating the heatmap.
        """

        if data == None:
            data = self.data
            title_data = "Original data"
        elif data == "data_income":
            data = self.data_income
            title_data = "Original + income data"

        if columns == None:
            columns = data.columns
            title_columns = "all columns"
        else:
            title_columns = f"subset of columns"

        corr_visualiser = EDA_Visualiser(data)
        title = f"Correlation_heatmap_{title_data}_{title_columns}"
        corr_visualiser.correlation_heatmap(columns=columns, title=title)

    def calc_vif(
        self, data: None | str = None, columns: None | list[str] = None
    ) -> None:
        """
        Calculate Variance Inflation Factor between specified variables in specified dataset.

        : param data: NoneType or str: Specifies which dataset to use. None is default and uses original dataset. Other options are data_income,
        : param columns: NoneType or list: List of columns to use for calculation.
        """
        if data == None:
            data = self.data
        elif data == "data_income":
            data = self.data_income

        if columns == None:
            columns = data.columns

        X = data[columns]
        vif = pd.DataFrame()
        vif["Feature"] = X.columns
        vif["VIF"] = [
            variance_inflation_factor(X.values, i) for i in range(len(X.columns))
        ]
        print(vif)
        vif.to_excel("./Results-Graphs/VIF.xlsx", index=False)

    def princomp(
        self,
        data: None | str = None,
        columns: None | list[str] = None,
        n_components: None | int = None,
    ) -> None:
        """
        Perform PCA for dimension reduction on the dataset with specified columns.

        : param data: NoneType or str: Specifies which dataset to use. None is default and uses original dataset. Other options are data_income,
        : param columns: list[string]: List of column names to include in PCA.
        : param n_components: int: Number of components to extract. If None (default), extract all components.
        """

        if data == None:
            data = self.data
            title_data = "Original_data"
        elif data == "data_income":
            data = self.data_income
            title_data = "Original_income data"

        if columns == None:
            columns = data.columns
            title_columns = "all_columns"
        else:
            title_columns = f"subset_columns"

        # Get columns and scale data since sklearn pca on covariance
        data_pca = data[columns]

        if n_components == None:
            if data_pca.shape[0] <= data_pca.shape[1]:
                n_components = data_pca.shape[0] - 1
            else:
                n_components = data_pca.shape[1]

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_pca)

        # Perform PCA
        pca = PCA(n_components=n_components)
        pca.fit(scaled_data)

        # Get explained variance ratios

        explained_variance_ratio = pca.explained_variance_ratio_

        explained_variance_df = pd.DataFrame(
            {
                "Principal Component": [
                    f"PC{i+1}" for i in range(len(explained_variance_ratio))
                ],
                "Explained Variance Ratio": explained_variance_ratio,
                "Cumulative Explained Variance": np.cumsum(explained_variance_ratio),
            }
        )

        explained_variance_df.to_excel(
            f"./Results-Graphs/PCA_{title_data}_{title_columns}_ExplainedVariance.xlsx",
            index=False,
        )
        print(explained_variance_df)

        # Get loadings

        loadings = pd.DataFrame(
            pca.components_.T * np.sqrt(pca.explained_variance_),
            columns=[f"PC{i+1}" for i in range(len(columns))],
            index=columns,
        )

        loadings.to_excel(
            f"./Results-Graphs/PCA_{title_data}_{title_columns}_Loadings.xlsx",
            index=True,
        )
        print(loadings)

        # Instantiate Visualiser and create useful plots for analysis

        pca_vis = PCA_Visualiser(pca=pca, data=data_pca)
        pca_vis.broken_stick()

        pca_vis.loadings(loadings=loadings)

        # Get PCA scores and store
        scores = pca.fit_transform(scaled_data)

        comp_columns = []
        for comp in range(n_components):
            comp_columns.append(f"PC{comp + 1}")

        scores_df = pd.DataFrame(scores, columns=comp_columns)
        scores_df["Price"] = data["Price"]

        pca_vis.correlation_components(scores=scores_df)

        # Add scores to data

        for comp in range(n_components):
            col_name = f"PC{comp + 1}"
            if title_data == "Original_data":
                self.data[col_name] = scores_df[col_name]
            elif title_data == "Original_income data":
                self.data_income[col_name] = scores_df[col_name]

        print(data.info())

    def get_modelling_X(self, columns: list[str]) -> None:
        """
        Define predictors to use for modelling purposes

        : param columns: list[string]: List of column names to retain for analysis.
        """
        self.modelling_data_X = self.data_income[columns].values
        print(
            f"Modelling predictor dataset created with shape: {self.modelling_data_X.shape}\nAnd characteristics:\n"
        )
        print(self.data_income[columns].info())

    def get_modelling_y(self, columns: list[str] = ["Price"]) -> None:
        """
        Define response to use for modelling purposes, default is Price.

        : param columns: list[string]: List of column names to retain for analysis.
        """
        self.modelling_data_y = self.data_income[columns].values
        print(
            f"Modelling response dataset created with shape: {self.modelling_data_y.shape}"
        )

    def calc_gower_dist(self) -> None:
        """
        Calculate the Gower distance between data points.
        """
        self.gowermat = gower_matrix(self.modelling_data_X)
        print(
            f"Calculated Gower distance\nDistance matrix shape: {self.gowermat.shape}"
        )

    def gower_train_test(self) -> None:
        """
        Split Gower distance matrix in training(0.8) and testing(0.2) set.
        """

        # Total number of points (matrix size is 10997 x 10997)
        n_points = self.gowermat.shape[0]

        seed = 42
        np.random.seed(seed)

        # Fraction of points to select
        train_fraction = 0.8

        # Generate all possible indices
        all_indices = np.arange(n_points)

        # Randomly shuffle the indices
        np.random.shuffle(all_indices)

        # Select 80% of the indices for training
        n_train = int(train_fraction * n_points)
        train_indices = all_indices[:n_train]

        # The remaining 20% are for testing
        test_indices = all_indices[n_train:]

        y = self.modelling_data_y
        self.y_train = y[train_indices]
        self.y_test = y[test_indices]

        D = self.gowermat[:, train_indices]
        self.X_train = D[train_indices]
        self.X_test = D[test_indices]

    def split_scale_train_test(self) -> None:
        """
        Get split of training(0.8) and testing(0.2) data and scale X.
        """

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.modelling_data_X, self.modelling_data_y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)

    def label_encode_cat(self, category: str) -> None:
        """
        Generate label encoding of categorical data.

        : param category: str: Name of category to encode.
        """
        new_col = category + "_labelencoded"
        self.data_income[new_col] = (
            self.data_income[category].astype("category").cat.codes
        )

    def onehot_encode_cat(self, category: str) -> None:
        """
        Generate one hot encoding of categorical data.

        : param category: str: Name of category to encode.
        """
        pd.get_dummies(self.data_income, columns=[category])
