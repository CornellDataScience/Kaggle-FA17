import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from model_super import *
from models import *
from ensembler import *

#note:  dropping more "useless" columns has decreased validation training time significantly

#Best validation score: 0.0518398510732 (Ridge Regression as decider, LGBM and XGBoost, one each)
#Other scores:  0.0519503944372 (Dropped DecisionTree and RandomForest)
#Other scores: 0.0520887540916 (K-fold validation with 250 rounds)
#Other scores: 0.0524806599371 (2nd XGBoost decreasd to 250 rounds + rf - 100 estimators)
#Other scores: 0.053413687092 (dropped bad columns)
#Other scores: 0.0534753444281 (added mean normalization)
#Other scores:  0.0539496646946
#Other scores:  0.0532557722982 (K-fold validation added with 500 rounds)

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
                         'propertycountylandusecode', 'buildingclasstypeid', 'decktypeid',
                         'hashottuborspa', 'poolcnt', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'storytypeid',
                         'fireplaceflag', 'assessmentyear', 'taxdelinquencyflag'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

categorical_cols = ['transaction_month', 'transaction_day', 'transaction_quarter', 'airconditioningtypeid',
                    'buildingqualitytypeid', 'fips', 'heatingorsystemtypeid', 'propertylandusetypeid', 'regionidcity',
                    'regionidcounty', 'regionidneighborhood','regionidzip', 'yearbuilt']

#perform mean normalization
num_changed = 0
#for column in x_train:
    #if column not in categorical_cols:
        #mean = x_train[column].mean()
        #stdev = x_train[column].std()
        #if stdev != 0:
            #x_train[column] = (x_train[column] - mean) / stdev
            #num_changed += 1
#print("number changed: ")
#print(num_changed)

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
    'reg_lambda': 0.8,
    'reg_alpha': 0.4,
    'base_score': y_mean,
    'eval_metric': 'mae',
    'nthread': 4,
    'random_state': 1,
    #'seed': 143,
    'silent': 1
}

#Lightgbm params
params_1 = {}
params_1['max_bin'] = 10
params_1['learning_rate'] = 0.0021 # shrinkage_rate
params_1['boosting_type'] = 'gbdt'
params_1['objective'] = 'regression'
params_1['metric'] = 'mae'          # or 'mae'
params_1['sub_feature'] = 0.345
params_1['bagging_fraction'] = 0.85 # sub_row
params_1['bagging_freq'] = 40
params_1['bagging_seed'] = 1
params_1['num_leaves'] = 512        # num_leaf
#params_1['min_data'] = 500         # min_data_in_leaf
params_1['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
params_1['verbose'] = 1
#params_1['feature_fraction_seed'] = 2
#params_1['bagging_seed'] = 3

# rf params
rf_params = {
    'n_estimators': 100,
    'max_depth': 8,
    'min_samples_split': 100,
    'min_samples_leaf': 30
}

#Extra Trees params
"""et_params = {
    'n_jobs': -1,
    'n_estimators': 500,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}"""

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
xgboost_1 = XGBoost(xgb_params, 500)
lgb_model_1 = LGBM(params_1)

base_models = [xgboost_1, lgb_model_1]
#create an ensembler with the base_models
#ensemble_model = Ensembler(base_models, XGBoost(xgb_params_2, 200))
#test different rounds again for xgboost
#train ensembler and save necessary values to create a heatmap
#first_layer_results = ensemble_model.train(x_train, y_train)
#create heatmap
#ensemble_model.heatmap(first_layer_results)

#predictions result same across all?  train issues?

############################################# Evaluate on validation set ###############################################

#ensemble_model.validation_accuracy(x_val, y_val)
#print('part 2')
ensemble_model_2 = Ensembler(base_models, RidgeModel())
first_layer_results_2 = ensemble_model_2.train(x_train, y_train)
ensemble_model_2.validation_accuracy(x_val, y_val)



########################################  Predict on Kaggle data and generate submission file ##########################

#ensemble_model.generateKaggleSubmission(sample, prop, train_columns)
