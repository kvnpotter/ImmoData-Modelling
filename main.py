# Imports

from classes.Preprocessor import DataPreprocessor
from classes.Modelling import Modeller
from classes.Visualiser import EDA_Visualiser, SHAP_Visualiser
from classes.Evaluation import Evaluator
import pickle


# Code

## Load dataset and add income/tax data

# dataset_1 = DataPreprocessor()
# dataset_1.load_data('./Data/Clean_data.csv')
# dataset_1.add_mean_income()

## View correlations between numerical predictor groups (property and tax data) and response variable (price)
## Select one tax data variable most correlated to price ; possible replacement for postal code
## Calculate VIF

# dataset_1.view_correlations(data= 'data_income',
#                            columns= ['N_Inhabitants', 'Tot_taxable_income', 'Mean_income_taxunit', 'Median_income_taxunit', 'Mean_income_inhabitant', 'Wealth_index', 'State_tax', 'Local_tax', 'Tot_tax', 'Price'])

# dataset_1.view_correlations(data= 'data_income',
#                            columns= ['Garden area', 'Surface of the land', 'Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Terrace area', 'Number of facades', 'Price'])

# dataset_1.calc_vif(data= 'data_income',
# columns= ['Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'])

## As per EDA proposition ; perform PCA on 4 most correlated predictor variables (property group) to price
## Possibly replace the original predictors by scores on PCA axes

# dataset_1.princomp(data= 'data_income',
#                   columns= ['Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'])

# Modelling

## Model 1 - KNN regression Price vs. most important predictors from EDA
## Without PCA - 'Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'
## Categoricals label encoded - 'PostalCodes', 'Subtype of property', 'State of the building'
## Train - test split 0.8-0.2
## Standard scaler
## GridSearch CV, Euclidean and cosine distance, uniform and inv. dist. weight
## 5 folds, neg RMSE scoring
## best [n_neighbors= 25, metric= 'euclidean', weights= 'distance']

## Label encode categorical variables, set X and y datasets, scale X and split train-test datasets

# dataset_1.label_encode_cat('Subtype of property')
# dataset_1.label_encode_cat('State of the building')
# dataset_1.label_encode_cat('PostalCodes')
# dataset_1.get_modelling_X(columns= ['PostalCodes_labelencoded', 'Subtype of property_labelencoded', "State of the building_labelencoded", 'Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'])
# dataset_1.get_modelling_y()
# dataset_1.split_scale_train_test()

## Create modeller object, set parameters, create model, evaluate, print to screen and write metrics to CSV

# CV_model = Modeller()
# CV_model.set_parameters(CV= True, distance= 'euclidean')
# CV_model.get_model(dataset_1.X_train_scaled, dataset_1.y_train)

# CV_model_eval = Evaluator(model= CV_model.model,
#                                X_train= dataset_1.X_train_scaled,
#                                 X_test= dataset_1.X_test_scaled,
#                                 y_train = dataset_1.y_train,
#                                 y_test= dataset_1.y_test)
# CV_model_eval.model_metrics('Raw_Numericals+Label_Encoding+PostalCode+CV+Euc_Cos')

## Model 2 - KNN regression Price vs. most important predictors from EDA
## Without PCA - 'Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'
## Categoricals as-is (Gower deals with automatically) - 'PostalCodes', 'Subtype of property', 'State of the building'
## Train - test split 0.8-0.2
## No scaling, based on distance matrix immediately
## GridSearch CV, Gower distance, uniform and inv. dist. weight
## 5 folds, neg RMSE scoring
## best [n_neighbors= 17, metric= 'precomputed', weights= 'distance']

# dataset_2 = DataPreprocessor()
# dataset_2.load_data('./Data/Clean_data.csv')
# dataset_2.add_mean_income()

# dataset_2.get_modelling_X(columns= ['PostalCodes', 'Subtype of property', "State of the building", 'Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'])
# dataset_2.get_modelling_y()

## Calculate Gower distance and split in training-testing data

# dataset_2.calc_gower_dist()
# dataset_2.gower_train_test()

# Gower_CV_model = Modeller()
# Gower_CV_model.set_parameters(distance= 'Gower')
# Gower_CV_model.get_model(dataset_2.X_train, dataset_2.y_train)

# Gower_CV_model_eval = Evaluator(model= Gower_CV_model.model,
#                                X_train= dataset_2.X_train,
#                                 X_test= dataset_2.X_test,
#                                 y_train = dataset_2.y_train,
#                                 y_test= dataset_2.y_test)
# Gower_CV_model_eval.model_metrics('Raw_Numericals+PostalCode+CV+Gow')

## Model 3 - KNN regression Price vs. most important predictors from EDA
## With PCA - 'Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'
## Categoricals label encoded - 'PostalCodes', 'Subtype of property', 'State of the building'
## Train - test split 0.8-0.2
## Standard scaler
## GridSearch CV, Euclidean and cosine distance, uniform and inv. dist. weight
## 5 folds, neg RMSE scoring
## best [n_neighbors= 25, metric= 'euclidean', weights= 'distance']

# dataset_3 = DataPreprocessor()
# dataset_3.load_data('./Data/Clean_data.csv')
# dataset_3.add_mean_income()

## As per EDA proposition ; perform PCA on 4 most correlated predictor variables (property group) to price
## Possibly replace the original predictors by scores on PCA axes

# dataset_3.princomp(data= 'data_income',
#                   columns= ['Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'])

# dataset_3.label_encode_cat('Subtype of property')
# dataset_3.label_encode_cat('State of the building')
# dataset_3.label_encode_cat('PostalCodes')
# dataset_3.get_modelling_X(columns= ['PostalCodes_labelencoded', 'Subtype of property_labelencoded', "State of the building_labelencoded", 'PC1', 'PC2'])
# dataset_3.get_modelling_y()
# dataset_3.split_scale_train_test()

# CV_model = Modeller()
# CV_model.set_parameters(CV= True, distance= 'euclidean')
# CV_model.get_model(dataset_3.X_train_scaled, dataset_3.y_train)

# CV_model_eval = Evaluator(model= CV_model.model,
#                                X_train= dataset_3.X_train_scaled,
#                                 X_test= dataset_3.X_test_scaled,
#                                 y_train = dataset_3.y_train,
#                                 y_test= dataset_3.y_test)
# CV_model_eval.model_metrics('PCA+Label_Encoding+PostalCode+CV+Euc_Cos')

## Model 4 - KNN regression Price vs. most important predictors from EDA
## With PCA - 'Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'
## Categoricals as-is (Gower deals with automatically) - 'PostalCodes', 'Subtype of property', 'State of the building'
## Train - test split 0.8-0.2
## No scaling, based on distance matrix immediately
## GridSearch CV, Gower distance, uniform and inv. dist. weight
## 5 folds, neg RMSE scoring
## best [n_neighbors= 29, metric= 'precomputed', weights= 'distance']

# dataset_4 = DataPreprocessor()
# dataset_4.load_data('./Data/Clean_data.csv')
# dataset_4.add_mean_income()

# dataset_4.princomp(data= 'data_income',
#                   columns= ['Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'])

# dataset_4.get_modelling_X(columns= ['PostalCodes', 'Subtype of property', "State of the building", 'PC1', 'PC2'])
# dataset_4.get_modelling_y()

# dataset_4.calc_gower_dist()
# dataset_4.gower_train_test()

# Gower_CV_model = Modeller()
# Gower_CV_model.set_parameters(distance= 'Gower')
# Gower_CV_model.get_model(dataset_4.X_train, dataset_4.y_train)

# Gower_CV_model_eval = Evaluator(model= Gower_CV_model.model,
#                                X_train= dataset_4.X_train,
#                                 X_test= dataset_4.X_test,
#                                 y_train = dataset_4.y_train,
#                                 y_test= dataset_4.y_test)
# Gower_CV_model_eval.model_metrics('PCA+PostalCode+CV+Gow')

## Model 5 - KNN regression Price vs. most important predictors from EDA
## Without PCA - 'Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'
## Replace postal codes with tax info
## Categoricals label encoded - 'Subtype of property', 'State of the building'
## Train - test split 0.8-0.2
## Standard scaler
## GridSearch CV, Euclidean and cosine distance, uniform and inv. dist. weight
## 5 folds, neg RMSE scoring
## best [n_neighbors= 23, metric= 'euclidean', weights= 'distance']

# dataset_5 = DataPreprocessor()
# dataset_5.load_data('./Data/Clean_data.csv')
# dataset_5.add_mean_income()

# dataset_5.label_encode_cat('Subtype of property')
# dataset_5.label_encode_cat('State of the building')

# dataset_5.get_modelling_X(columns= ['Mean_income_taxunit', 'Subtype of property_labelencoded', "State of the building_labelencoded", 'Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'])
# dataset_5.get_modelling_y()
# dataset_5.split_scale_train_test()

# CV_model = Modeller()
# CV_model.set_parameters(CV= True, distance= 'euclidean')
# CV_model.get_model(dataset_5.X_train_scaled, dataset_5.y_train)

# CV_model_eval = Evaluator(model= CV_model.model,
#                                X_train= dataset_5.X_train_scaled,
#                                 X_test= dataset_5.X_test_scaled,
#                                 y_train = dataset_5.y_train,
#                                 y_test= dataset_5.y_test)
# CV_model_eval.model_metrics('Raw_Numericals+Label_Encoding+TaxVar+CV+Euc_Cos')

## Model 6 - KNN regression Price vs. most important predictors from EDA
## Without PCA - 'Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'
## Replace postal codes with tax variable
## Categoricals as-is (Gower deals with automatically) - 'Subtype of property', 'State of the building'
## Train - test split 0.8-0.2
## No scaling, based on distance matrix immediately
## GridSearch CV, Gower distance, uniform and inv. dist. weight
## 5 folds, neg RMSE scoring
## best [n_neighbors= 17, metric= 'precomputed', weights= 'distance']

# dataset_6 = DataPreprocessor()
# dataset_6.load_data('./Data/Clean_data.csv')
# dataset_6.add_mean_income()

# dataset_6.get_modelling_X(columns= ['Mean_income_taxunit', 'Subtype of property', "State of the building", 'Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'])
# dataset_6.get_modelling_y()

# dataset_6.calc_gower_dist()
# dataset_6.gower_train_test()

# Gower_CV_model = Modeller()
# Gower_CV_model.set_parameters(distance= 'Gower')
# Gower_CV_model.get_model(dataset_6.X_train, dataset_6.y_train)

# Gower_CV_model_eval = Evaluator(model= Gower_CV_model.model,
#                                X_train= dataset_6.X_train,
#                                 X_test= dataset_6.X_test,
#                                 y_train = dataset_6.y_train,
#                                 y_test= dataset_6.y_test)
# Gower_CV_model_eval.model_metrics('Raw_Numericals+TaxVar+CV+Gow')

## Model 7 - KNN regression Price vs. most important predictors from EDA
## With PCA - 'Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'
## Replace postal codes with tax info
## Categoricals label encoded - 'Subtype of property', 'State of the building'
## Train - test split 0.8-0.2
## Standard scaler
## GridSearch CV, Euclidean and cosine distance, uniform and inv. dist. weight
## 5 folds, neg RMSE scoring
## best [n_neighbors= 26, metric= 'euclidean', weights= 'distance']

# dataset_7 = DataPreprocessor()
# dataset_7.load_data('./Data/Clean_data.csv')
# dataset_7.add_mean_income()

# dataset_7.label_encode_cat('Subtype of property')
# dataset_7.label_encode_cat('State of the building')

# dataset_7.princomp(data= 'data_income',
#                   columns= ['Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'])


# dataset_7.get_modelling_X(columns= ['Mean_income_taxunit', 'Subtype of property_labelencoded', "State of the building_labelencoded", 'PC1', 'PC2'])
# dataset_7.get_modelling_y()
# dataset_7.split_scale_train_test()

# CV_model = Modeller()
# CV_model.set_parameters(CV= True, distance= 'euclidean')
# CV_model.get_model(dataset_7.X_train_scaled, dataset_7.y_train)

# CV_model_eval = Evaluator(model= CV_model.model,
#                                X_train= dataset_7.X_train_scaled,
#                                 X_test= dataset_7.X_test_scaled,
#                                 y_train = dataset_7.y_train,
#                                 y_test= dataset_7.y_test)
# CV_model_eval.model_metrics('PCA+Label_Encoding+TaxVar+CV+Euc_Cos')

## Model 8 - KNN regression Price vs. most important predictors from EDA
## With PCA - 'Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'
## Replace postal codes with tax variable
## Categoricals as-is (Gower deals with automatically) - 'Subtype of property', 'State of the building'
## Train - test split 0.8-0.2
## No scaling, based on distance matrix immediately
## GridSearch CV, Gower distance, uniform and inv. dist. weight
## 5 folds, neg RMSE scoring
## best [n_neighbors= 20, metric= 'precomputed', weights= 'distance']

# dataset_8 = DataPreprocessor()
# dataset_8.load_data('./Data/Clean_data.csv')
# dataset_8.add_mean_income()

# dataset_8.princomp(data= 'data_income',
#                   columns= ['Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'])

# dataset_8.get_modelling_X(columns= ['Mean_income_taxunit', 'Subtype of property', "State of the building", 'PC1', 'PC2'])
# dataset_8.get_modelling_y()

# dataset_8.calc_gower_dist()
# dataset_8.gower_train_test()

# Gower_CV_model = Modeller()
# Gower_CV_model.set_parameters(distance= 'Gower')
# Gower_CV_model.get_model(dataset_8.X_train, dataset_8.y_train)

# Gower_CV_model_eval = Evaluator(model= Gower_CV_model.model,
#                                X_train= dataset_8.X_train,
#                                 X_test= dataset_8.X_test,
#                                 y_train = dataset_8.y_train,
#                                 y_test= dataset_8.y_test)
# Gower_CV_model_eval.model_metrics('PCA+TaxVar+CV+Gow')

## Model 2 - Selected as best model - perform SHAP analysis
## Without PCA - 'Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'
## Categoricals as-is (Gower deals with automatically) - 'PostalCodes', 'Subtype of property', 'State of the building'
## Train - test split 0.8-0.2
## No scaling, based on distance matrix immediately
## GridSearch CV, Gower distance, uniform and inv. dist. weight
## 5 folds, neg RMSE scoring
## best [n_neighbors= 17, metric= 'precomputed', weights= 'distance']

dataset_2 = DataPreprocessor()
dataset_2.load_data("./Data/Clean_data.csv")
dataset_2.add_mean_income()

dataset_2.get_modelling_X(
    columns=[
        "PostalCodes",
        "Subtype of property",
        "State of the building",
        "Surface area of the plot of land",
        "Number of rooms",
        "Living Area",
        "Number of facades",
    ]
)
dataset_2.get_modelling_y()

dataset_2.calc_gower_dist()
dataset_2.gower_train_test()

Gower_CV_model = Modeller()
Gower_CV_model.set_parameters(distance="Gower")
Gower_CV_model.get_model(dataset_2.X_train, dataset_2.y_train)

## Save best model in pickle file

with open("./Results-Graphs/best_knn_model.pkl", "wb") as f:
    pickle.dump(Gower_CV_model.model, f)
