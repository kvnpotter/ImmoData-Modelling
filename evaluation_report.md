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


a code snippet of the model instantiation (to see which model it is, what parameters,...)
MAE (on training/test)
RMSE (on training/test)
R2 (on training/test)
MAPE (on training/test)
sMAPE (on training/test)
The list of features you've used and how you got it (to quickly understand if you've done data leakage)
Accuracy computing procedure (on a test set? What split %, 80/20, 90/10, 50/50? k-fold cross?)
Efficiency (training and inference time). The fastest the best (sustainability).
a quick presentation of the final dataset (how many records, did you merge some datasets together? did you scrape again? what cleaning step you've done, scaling, encoding, cross-validation.. No need of visuals, just bullet points)
