# Imports

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from gower import gower_matrix

from classes.Visualiser import PCA_Visualiser

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

    def load_data(self, data_path: str) -> None:
        """
        Load dataset, assign correct types and store in data attribute.

        : param data_path: str: String containing data path.
        """
        data = pd.read_csv(data_path)

        data.drop(columns=['Unnamed: 0', 'id'], inplace=True)
        data['Locality'] = data['Locality'].astype('str')
        data['Fully equipped kitchen'] = data['Fully equipped kitchen'].astype('str')
        data['Fireplace'] = data['Fireplace'].astype('str')
        data['Terrace'] = data['Terrace'].astype('str')
        data['Garden'] = data['Garden'].astype('str')
        data['Swimming pool'] = data['Swimming pool'].astype('str')
        data['Furnished'] = data['Furnished'].astype('str')
        data['Number of rooms'] = data['Number of rooms'].astype('int64')
        data['Number of facades'] = data['Number of facades'].astype('int64')
        
        self.data = data
        print(f"Dataset loaded!\nThe loaded (original) dataset characteristics:\n")
        print(self.data.info())


    def add_mean_income(self) -> None:
        """
        Load datasets to join tables, and add mean income per locality.
        """
        # Import and prepare dataset to join Code INS to Postal code
        data_insee = pd.read_csv('./Data/INSEE_PostCode.csv', encoding='latin-1')
        subset_columns = data_insee.columns[6:]
        data_insee["PostalCodes"] = data_insee[subset_columns].apply(lambda row: row.dropna().tolist(), axis=1)
        data_insee.drop(columns= data_insee.columns[6:22], inplace= True)
        

        # Import and prepare dataset from SPF Finances to join on Code INS
        data_fin = pd.read_csv('./Data/SPF_FIN_Stat.csv', encoding= 'latin-1')

        # Merge two datasets on Code INS, keeping only records that are present in both tables
        # Rows without postal code, or without financial data are not interesting for final join with data
        data_fin_postcode = pd.merge(data_fin, data_insee, how="inner", on="Code INS")

        # Unpack/explode lists of post codes to obtain dataset with one row of info per postal code
        data_fin_postcode_exploded = data_fin_postcode.explode('PostalCodes')

        # Convert post codes to str type and join data to original dataset
        data_fin_postcode_exploded['PostalCodes'] = data_fin_postcode_exploded['PostalCodes'].astype('int').astype('str')
        data_postcodes = pd.merge(data_fin_postcode_exploded, self.data, how='inner', left_on='PostalCodes', right_on='Locality')

        # Drop duplicate columns, rename columns
        data_postcodes.drop(columns= 'Entités administratives_x', inplace= True)
        data_postcodes.drop(columns= 'Locality', inplace= True)
        data_postcodes.drop(columns= 'Code INS', inplace= True)
        data_postcodes.rename(columns={'Entités administratives_y': 'Locality'}, inplace=True)
        data_postcodes.rename(columns={"Nombre d'habitants": 'N_Inhabitants'}, inplace=True)
        data_postcodes.rename(columns={"Revenu total net imposable": 'Tot_taxable_income'}, inplace=True)
        data_postcodes.rename(columns={"Revenu moyen par déclaration": 'Mean_income_taxunit'}, inplace=True)
        data_postcodes.rename(columns={"Revenu médian par déclaration": 'Median_income_taxunit'}, inplace=True)
        data_postcodes.rename(columns={"Revenu moyen par habitant": 'Mean_income_inhabitant'}, inplace=True)
        data_postcodes.rename(columns={"Indice de richesse": 'Wealth_index'}, inplace=True)
        data_postcodes.rename(columns={"Impôt d'Etat": 'State_tax'}, inplace=True)
        data_postcodes.rename(columns={"Taxes communales et d'agglomération": 'Local_tax'}, inplace=True)
        data_postcodes.rename(columns={"Impôt total": 'Tot_tax'}, inplace=True)
        data_postcodes.rename(columns={"Langue": 'Language'}, inplace=True)
        data_postcodes.rename(columns={"Région": 'Region'}, inplace=True)
        data_postcodes.rename(columns={"Arrondissement": 'District'}, inplace=True)
        data_postcodes.rename(columns={"Indice de richesse": 'Wealth_index'}, inplace=True)
        data_postcodes.rename(columns={"Indice de richesse": 'Wealth_index'}, inplace=True)
        data_postcodes.rename(columns={"Indice de richesse": 'Wealth_index'}, inplace=True)
        data_postcodes.rename(columns={"Indice de richesse": 'Wealth_index'}, inplace=True)

        # clean data types
        columns_to_convert = ["N_Inhabitants", "Tot_taxable_income", "State_tax", "Local_tax", "Tot_tax"]
        data_postcodes[columns_to_convert] = data_postcodes[columns_to_convert].apply(lambda col: col.str.replace('.', '', regex=False)).astype(float)
        data_postcodes['N_Inhabitants'] = data_postcodes['N_Inhabitants'].astype(int)
        data_postcodes['Wealth_index'] = data_postcodes['Wealth_index'].astype(float)

        self.data_income = data_postcodes
        print(f"Data from SPF Finances added to dataset!\nCharacteristics of complete dataset:\n")
        print(self.data_income.info())

    def princomp(self, columns: list[str], n_components: int = 2) -> None:
        """
        Perform PCA for dimension reduction on the dataset with specified columns.

        : param columns: list[string]: List of column names to include in PCA.
        : param n_components: int: Number of components to extract.
        """
        #n_components = 

        # Get columns and scale data since sklearn pca on covariance
        data_pca = self.data_income[columns]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_pca)

        # Perform PCA
        pca= PCA()
        pca.fit(scaled_data)

        # Instantiate Visualiser and create useful plots for analysis

        pca_vis = PCA_Visualiser()

    def get_modelling_X(self, columns: list[str]) -> None:
        """
        Define predictors to use for modelling purposes

        : param columns: list[string]: List of column names to retain for analysis.
        """
        self.modelling_data_X = self.data_income[columns].values
        print(f"Modelling predictor dataset created with shape: {self.modelling_data_X.shape}\nAnd characteristics:\n")
        print(self.modelling_data_X.info())

    def get_modelling_y(self, columns: list[str]= ['Price']) -> None:
        """
        Define response to use for modelling purposes, default is Price.

        : param columns: list[string]: List of column names to retain for analysis.
        """
        self.modelling_data_y = self.data_income[columns].values
        print(f"Modelling response dataset created with shape: {self.modelling_data_y.shape}")

    def calc_gower_dist(self) -> None:
        """
        Calculate the Gower distance between data points.
        """
        self.gowermat = gower_matrix(self.modelling_data_X)
        print(f"Calculated Gower distance\nDistance matrix shape: {self.gowermat.shape}")