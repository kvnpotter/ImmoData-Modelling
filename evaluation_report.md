# Evaluation for the selected model

## 1. Code snippet of model instantiation

```python
# Set up preprocessor, load data and add tax data
dataset_2 = DataPreprocessor()
dataset_2.load_data('./Data/Clean_data.csv')
dataset_2.add_mean_income()

# Prepare X and y datasets
dataset_2.get_modelling_X(columns= ['PostalCodes', 'Subtype of property', "State of the building", 'Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades'])
dataset_2.get_modelling_y()

## Calculate Gower distance and split in training-testing data

dataset_2.calc_gower_dist()
dataset_2.gower_train_test()

# Prepare modeller, get model parameters based on input, create model and evaluate using evaluator
Gower_CV_model = Modeller()
Gower_CV_model.set_parameters(distance= 'Gower')
Gower_CV_model.get_model(dataset_2.X_train, dataset_2.y_train)

Gower_CV_model_eval = Evaluator(model= Gower_CV_model.model,
                                X_train= dataset_2.X_train,
                                 X_test= dataset_2.X_test,
                                 y_train = dataset_2.y_train,
                                 y_test= dataset_2.y_test)
Gower_CV_model_eval.model_metrics('Raw_Numericals+PostalCode+CV+Gow')
```

## 2. Model metrics

![evaluation_metrics](./Results-Graphs/table.JPG)

## 3. Used features

The features used to predict price include the following:
- Postal code (obj) : post code in string format (not encoded as categorical because used Gower distance which automatically deals with this)
- Subtype of property (obj) : idem above
- State of the building (obj) : idem above
- 'Surface area of the plot of land', 'Number of rooms', 'Living Area', 'Number of facades' (float and int) : raw data from ImmoWeb, numerical data was not scaled since a Gower distance matrix was used in final modelling. Gower distance calculates distances feature by feature, therefore eliminating issues with scaling.

## 4. Accuracy computing 

- Splitting data in 0.8 - 0.2 training - test sets
- Using 5 fold CV gridsearch for hyperparameter tuning

## 5. Efficiency

Not evaluated due to time constraints

## 6. Dataset

The final dataset includes 10278 observations drawn from ImmoWeb. 
In addition, tax/income data was added, but not used in the final model

## 7. Feature importance - SHAP

Not added due to time constraints

## 8. Remarks

From the research, modelling and selection, following remarks can be made:
- The best model was chosen based on R2, MAE (most interesting for final estimation use of the model) and RMSE
- Using PCA to decorrelate variables does not seem to strongly affect the model ; the choice was made to use the raw data to allow beter explanation of the variables included
- USing Gower distance appears to slightly improve the model in some cases
- KNN regression does not seem highy adapted to price prediction in this case
- Distance metrics degrade with increasing number of predictors (also leading to the choice of using label encoding rather than one hot encoding)
- High number of categories and class imbalance should be addressed in categorical predictors
- This could also lead to more outliers to resolve ; may give errors in prediction and/or extrapolation when the final model is deployed
- Increased training data could lead to better model performance
- The model appears to suffer from overfitting, but CV was already used to minimiz this
- The postal code does not seem an ideal predictor variable, however replacing it by tax/income data seems to lead to a slightly less reliable model. In deployment, if a user should enter a postalcode unknown to the model this will lead to issues, especially if the postal code is an important predictor (SHAP should be carried out).

