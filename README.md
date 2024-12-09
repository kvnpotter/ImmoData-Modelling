# **ImmoData - Modelling real estate data for price prediction usin KNN regression**

[Introduction](#Introduction)    |    [Description](#Description)    |    [Installation-Environment setup](#Installation-Environment-setup)    |    [Usage](#Usage)    |    [Contributors](#Contributors)    |    [Timeline](#Timeline)

## **Introduction**

This repo contains the results of the third project in a series aimed at completing a data science workflow from start (data collection) to finish (modelling using machine learning) during my AI and Data science bootcamp training at BeCode (Brussels, Belgium). The final goal is to create a machine learning model capable of predicting real estate prices in Belgium.

The specific aims for this project are to:
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
## Exploring real estate data

Following steps were taken to clean the data: 
- Duplicates handled (complete row duplicates, listings posted more than once on the website)
- Address data type constraints
- Identify patterns in missingness and address missing values with logical choices first
- Verify value constraints for categorical variables
- Remove outliers based on response variable, assuming differences in distribution for houses and apartments
- Deal with outliers in predictor space using multivariate method (DBSCAN)
- Impute remaining missing values (after outlier removal to ensure no bias introduced) using KNN

Packages used:
- pandas
- numpy
- matplotlib
- seaborn
- missingno
- sklearn

Following data cleaning, EDA was carried out according to following steps:
- Verification of distributions and relationships between numerical data variables
- Correlations between numerical variables - including response variable
- Verification of multicollinearity with **V**ariance **I**nflation **F**actor
- Attempt at dimension reduction utilising **P**rincipal **C**omponent **A**nalysis
- Identifying relationships between categorical predictors and response (Kruskal-Wallis and Mann-Whitney-U + example of post-hoc testing)
- Associations between categorical predictors (Χ squared)

Packages used:
- pandas
- numpy
- matplotlib
- seaborn
- sklearn
- scipy
- statsmodels

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

The repository contains following notebooks outlining the steps taken
- DataCleaningKevin : All steps performed to clean the dataset
- DataAnalysisKevin : All steps performed to analyse the dataset
- DataAnalysisMaxim : Number qualitative/quantitative variables, transformation categorical - numerical, test effect of transformation method on model, percentage missing values, spatial analysis
- correlation : Correlation and data visualization - Fatemeh

In addition, the full, and cleaned, datasets can be found in the Data directory.
When possible, tables and graphs from analysis results were added to the Results directory.


# Contributors 
This project was completed by:
1. [Kevin](https://github.com/kvnpotter)
   - Data cleaning
   - Distributions, normality, correlations, dimension reduction, differences in median price between groups
3. [Maxim](https://github.com/MaximSchuermans)
   - Transformation types from categorical to numerical values and test
   - Missing values
   - Spatial analysis of property price in Belgium
5. [Fatemeh](https://github.com/Fatemeh992)
   - Correlation and data visualization
# **Timeline**

Start project: 19/11/2024 09:30
End project, with presentation of results: 22/11/2024 12:30


