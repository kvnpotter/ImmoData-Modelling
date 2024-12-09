# **ImmoData - Modelling real estate data for price prediction using KNN regression**

[Introduction](#Introduction)    |    [Description](#Description)    |    [Installation-Environment setup](#Installation-Environment-setup)    |    [Usage](#Usage)    |    [Contributors](#Contributors)    |    [Timeline](#Timeline)

## **Introduction**

This repo contains the results of the third project in a series aimed at completing a data science workflow from start (data collection) to finish (modelling using machine learning) during my AI and Data science bootcamp training at BeCode (Brussels, Belgium). The final goal is to create a machine learning model capable of predicting real estate prices in Belgium.

The specific aims for this project are :
1. Being able to apply a regression model in a real context - apply the assigned model, in this case : KNN regression
2. Preprocessing data for machine learning

Specifications for the final dataset are:
- Handling NaN
- Handling categorical variables
- Select features + possible feature engineering
- Remove features resulting in strong multicollinearity
- Dividing dataset in training and testing sets
- Apply the assigned model (KNN regression) to the data
- Perform model evaluation

## **Description**

The following part of the README focuses on structure of the code and how modelling was carried out

## Data preparation and feature selection

All steps for data preparation and feature selection are taken by instantiating a Preprocessor object and running appropriate methods.
Methods for instances of Preprocessor class allow to load data, add income/tax variables, label and one-hot encode categorical variables, view correlations, calculate VIF, perform PCA (with appropriate visualisations), get datasets for X and y ready for modelling, calculate Gower distances and splitting + scaling the datasets in training and testing sets respectively.

Many preparation steps for feature selection and/or engineering were taken previously during the EDA part of this project.
Most notably, strongly correlated predictor variables, which were most correlated to the target, were decorrelated using PCA.
Additionally, the dataset contained no remaining missing values, etc. 

To replace the raw postal code data scraped from ImmoWeb (assuming this variable does not confer much information to the model, and does not easily allow extrapolation), data pertaining to tax information (at locality scale) was obtained from the Belgian government. Since many of the variables were correlated, only one (most strongly associated to price) was retained for model creation and testing.

## Modelling real estate data

Modelling occurs by instantiating a Modeller object. Methods available for this class allow to model the data using KNN regression and a combination of following parameters:
- n_neighbors : varying number of nearest neighbors to take into account
- metric : which distance metric to use for NN calculation (Euclidean, cosine or Gower)
- weights : which weighting to attribute to neighbors (uniform, inverste distance)

Methods available for Modeller objects allow setting model parameters (including adding cross-validation grid search or not), getting the model (a method permitting to send the data to the appropriate model, with or without CV gridsearch, based on selected parameters).

CV gridsearch, using 5 CV folds and based on RMSE scoring was used for hyperparameter tuning.
Gower distance was included in modelling, despite not being available natively in sklearn, due to the fact that Gower distance is assumed to be better suited to datasets mixing categorical and numerical data.

## Model evaluation

For model evaluation, an object of Evaluator class is instantiated, allowing to calculate, and store, model evaluation metrics such as R2, MAE, RMSE, MAPE, etc.

The choice was made to exclude R2 adjusted, in favour of simple R2 since the number of observations in the dataset is much larger than the number of features.

## Visualisation

A number of visualiser classes were constructed to group all functionality pertaining to the building of graphs.

## Main

The main script contains all iterations of modelling (8 total) on a combination of parameters (using CV gridsearch) and different combinations of predictors.
In particular:
- Using the original postal code data vs. replacing it with tax data
- Using the original (correlated) predictors at property level vs. observation scores on the first two axes of PCA on these variables

The results of each iteration were automatically recorded in the ./Results-Graphs/model_metrics.csv file

Finally, the script stores the best model (see evalutation_report.md) as a pickle file.

   ## **Installation-Environment setup**

You can create a virtual environment for the script using venv.
```shell
python -m venv C:\path\to\new\virtual\environment
```

Or using conda.
```shell
conda create --name <my-env>
conda activate <my-env>
```

Included in the repository is a cross-platform environment.yml file, allowing to create a copy of the one used for this project. The environment name is given in the first line of the file.
```shell
conda env create -f environment.yml
conda activate wikipedia_scraper_env
conda env list #verify the environment was installed correctly
```

## **Usage**

The repository contains following files and directories
- Model and Testing jupyter notebooks : Notebooks detailing modelling from loading the data to calculating evaluation metrics (to be completed)
- main.py : Main script, as described above, performing modelling, recording evaluation metrics and pickling the best model.
- (environment, license and readme files)
- test_scripts directory : containing an attempt at writing a function for Gower distance calculation (function from the gower package used in final modelling)
- Data directory : contains clean property data obtained from EDA project and income/tax data obtained from the belgian government (FOD Financiën/SPF Finances)
- Results-Graphs directory : contains all outputs from visualisation and table creation methods, including evaluation metrics for 8 models
- classes : contains separate modules for data preprocessing, modelling, evaluation and visualisation
- evaluation_report.md : a report evaluating the selected model

# Contributors 
This project was completed by:
1. [Kevin](https://github.com/kvnpotter)

# **Timeline**

Start project: 2/12/2024 09:30
End project: 9/12/2024 16:30


