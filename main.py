# Imports

from classes.Preprocessor import DataPreprocessor
from classes.Modelling import Modeller
from classes.Visualiser import EDA_Visualiser




# Code

dataset_1 = DataPreprocessor()
dataset_1.load_data('./Data/Clean_data.csv')
dataset_1.add_mean_income()

correlations_finance = EDA_Visualiser(data= dataset_1.data_income)
correlations_finance.correlation_heatmap(columns= ['N_Inhabitants', 'Tot_taxable_income', 'Mean_income_taxunit',
       'Median_income_taxunit', 'Mean_income_inhabitant', 'Wealth_index',
       'State_tax', 'Local_tax', 'Tot_tax', 'Price'], title= "correlations_finance")

correlations_property = EDA_Visualiser(data= dataset_1.data_income)
correlations_property.correlation_heatmap(columns= ['Garden area', 'Surface of the land', 'Surface area of the plot of land', 'Price', 'Number of rooms', 'Living Area', 'Terrace area', 'Number of facades'], title= "correlations_property")
       






       
       

