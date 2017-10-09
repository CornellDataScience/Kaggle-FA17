import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from tqdm import tqdm

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
#0.0671924136962 (4 models, Elastic Net with alpha of 0.57)

################################################ Load Data #############################################################

train_df = pd.read_csv('/Users/kevinluo/Desktop/zillow_data/train_2016.csv', parse_dates=['transactiondate'],
                       low_memory=False)
test_df = pd.read_csv('/Users/kevinluo/Desktop/zillow_data/submission.csv', low_memory=False)
properties = pd.read_csv('/Users/kevinluo/Desktop/zillow_data/properties_2016.csv', low_memory=False)
# field is named differently in submission
test_df['parcelid'] = test_df['ParcelId']

############################################### Process Data ###########################################################

# similar to the1owl
def add_date_features(df):
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = df["transactiondate"].dt.month
    df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = df["transactiondate"].dt.quarter
    df.drop(["transactiondate"], inplace=True, axis=1)
    return df

train_df = add_date_features(train_df)
train_df = train_df.merge(properties, how='left', on='parcelid')
test_df = test_df.merge(properties, how='left', on='parcelid')

missing_perc_thresh = 0.98
exclude_missing = []
num_rows = train_df.shape[0]
for c in train_df.columns:
    num_missing = train_df[c].isnull().sum()
    if num_missing == 0:
        continue
    missing_frac = num_missing / float(num_rows)
    if missing_frac > missing_perc_thresh:
        exclude_missing.append(c)
print("We exclude: %s" % exclude_missing)
print(len(exclude_missing))

# exclude where we only have one unique value :D
exclude_unique = []
for c in train_df.columns:
    num_uniques = len(train_df[c].unique())
    if train_df[c].isnull().sum() != 0:
        num_uniques -= 1
    if num_uniques == 1:
        exclude_unique.append(c)
print("We exclude: %s" % exclude_unique)
print(len(exclude_unique))

exclude_other = ['parcelid', 'logerror']  # for indexing/training only
# do not know what this is LARS, 'SHCG' 'COR2YY' 'LNR2RPD-R3' ?!?
exclude_other.append('propertyzoningdesc')
train_features = []
for c in train_df.columns:
    if c not in exclude_missing \
       and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)
print("We use these for training: %s" % train_features)
print(len(train_features))

cat_feature_inds = []
cat_unique_thresh = 1000
for i, c in enumerate(train_features):
    num_uniques = len(train_df[c].unique())
    if num_uniques < cat_unique_thresh \
            and not 'sqft' in c \
            and not 'cnt' in c \
            and not 'nbr' in c \
            and not 'number' in c:
        cat_feature_inds.append(i)

print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])

# some out of range int is a good choice
train_df.fillna(-1, inplace=True)
test_df.fillna(-1, inplace=True)
train_df = train_df[train_df.logerror > -0.4]
train_df = train_df[train_df.logerror < 0.4]

x_train = train_df[train_features]

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

#Simplify Data Types
for c, dtype in zip(x_train.columns, x_train.dtypes):
	if dtype == np.float64:
		x_train[c] = x_train[c].astype(np.float32)

y_train = train_df['logerror'].values

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=27000)
print("validation shape: ")
print(y_val.shape)

#test_df['transactiondate'] = pd.Timestamp('2016-12-01')  # Dummy
#test_df = add_date_features(test_df)
#X_test = test_df[train_features]
#print(X_test.shape)

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
    'n_jobs': 4,
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
params_1['verbose'] = 0
#params_1['feature_fraction_seed'] = 2
#params_1['bagging_seed'] = 3

#CatBoost params
catboost_params = {
    'iterations': 200,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'loss_function': 'MAE',
    'eval_metric': 'MAE'
}

# rf params
rf_params = {
    'n_estimators': 100,
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

elastic_net_params = {
    'alpha': 0.57541585223123481
}

#Initialize all of the baseline models
xgboost_1 = XGBoost(xgb_params, 500)
lgb_model_1 = LGBM(params_1)
catboost = CatBoostModel(catboost_params, cat_feature_inds)
randomForest = RandomForest(rf_params)

base_models = [randomForest, xgboost_1, catboost, lgb_model_1]
#create an ensembler with the base_models
ensemble_model = Ensembler(base_models, ElasticNetModel(elastic_net_params))

#train ensembler and save necessary values to create a heatmap
first_layer_results = ensemble_model.train(x_train, y_train)
#create heatmap
ensemble_model.heatmap(first_layer_results)

#0.0526758692713


############################################# Evaluate on validation set ###############################################

ensemble_model.validation_accuracy(x_val, y_val)

########################################  Predict on Kaggle data and generate submission file ##########################

#ensemble_model.generateKaggleSubmission(sample, prop, train_columns)"""