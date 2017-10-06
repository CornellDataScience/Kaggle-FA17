import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from model_super import *
from models import *
from ensembler import *

#normalize and standardize data?  some categories have extremely high relative values
#Best validation score:  0.0535038790343

################################################ Load Data #############################################################

print('Loading data ...')

train = pd.read_csv('/Users/kevinluo/Desktop/zillow_data/train_2016.csv')
prop = pd.read_csv('/Users/kevinluo/Desktop/zillow_data/properties_2016.csv')
sample = pd.read_csv('/Users/kevinluo/Desktop/zillow_data/submission.csv')

############################################### Process Data ###########################################################

#Deal with NA Values
for c in prop.columns:
    prop[c]=prop[c].fillna(-1)

print('Binding to float32')

#Simplify Data Types
for c, dtype in zip(prop.columns, prop.dtypes):
	if dtype == np.float64:
		prop[c] = prop[c].astype(np.float32)

print('Creating training set ...')

df_train = train.merge(prop, how='left', on='parcelid')

#Remove outliers
df_train = df_train[df_train.logerror > -0.4]
df_train = df_train[df_train.logerror < 0.4]

#Drop bad columns
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
                         'propertycountylandusecode'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=27000)

######################################################## Training ######################################################

print('Training ...')

y_mean = y_train.mean()
#XGBoost params
xgb_params = {
    'eta': 0.035,
    'max_depth': 6,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'colsample_bytree': 0.9,
    'eval_metric': 'mae',
    'lambda': 0.8,
    'alpha': 0.4,
    'base_score': y_mean,
    'seed': 143,
    'silent': 1
}

# rf params
rf_params = {
    'n_estimators': 50,
    'max_depth': 8,
    'min_samples_split': 100,
    'min_samples_leaf': 30
}

#Extra Trees params
et_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 400,
    'learning_rate' : 0.75
}

#Second Layer XGBoost parameters
xgb_params_2 = {
            'eta': 0.037,
            'max_depth': 5,
            'subsample': 0.80,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'lambda': 0.8,
            'alpha': 0.4,
            'silent': 1
        }

#Initialize all of the baseline models
random_forest = RandomForest(rf_params)
extra_trees = ExtraTrees(et_params)
ada_boost = AdaBoost(ada_params)
decision_tree = DecisionTree({})
xgboost = XGBoost(xgb_params)

base_models = [random_forest, extra_trees, ada_boost, decision_tree, xgboost]
#create an ensembler with the base_models
#ensemble_model = Ensembler(base_models, LinearRegressionModel({}))
ensemble_model = Ensembler(base_models, XGBoost(xgb_params_2))
#train ensembler and save necessary values to create a heatmap
first_layer_results = ensemble_model.train(x_train, y_train)
#create heatmap
ensemble_model.heatmap(first_layer_results)

############################################# Evaluate on validation set ###############################################

ensemble_model.validation_accuracy(x_val, y_val)

########################################  Predict on Kaggle data and generate submission file ##########################

#ensemble_model.generateKaggleSubmission(sample, prop, train_columns)
