from mlens.ensemble import SuperLearner

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import ElasticNet
from tqdm import tqdm
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy.stats import uniform, randint
from mlens.preprocessing import EnsembleTransformer
from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator

from model_super import *
from models import *
from ensembler import *

#results of iteration 1 (all models):  example.csv for params, 0.052445
#results of iteration 2 (three models):  iteration2.csv for params, 0.052109

def evaluateSecondLayer():
    in_layer = EnsembleTransformer()
    print("adding base learners to transformer")
    in_layer.add('stack', base_learners)

    preprocess = [in_layer]
    print("creating scorer")
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    evl = Evaluator(scorer, cv=4, verbose=1)
    print("fitting evaluator")
    evl.fit(x_train.values,
        y_train.values,
        meta_learners,
        param_dicts,
        preprocessing={'meta': preprocess},
        n_iter=40                            # bump this up to do a larger grid search
       )

    table = pd.DataFrame(evl.summary)
    table.to_html('iteration4.html')
    table.to_csv('iteration4.csv', index=False, header=False, sep='\t')

#LOADING DATA

train_df = pd.read_csv('/Users/kevinluo/Desktop/zillow_data/train_2016.csv', parse_dates=['transactiondate'],
                       low_memory=False)
test_df = pd.read_csv('/Users/kevinluo/Desktop/zillow_data/submission.csv', low_memory=False)
properties = pd.read_csv('/Users/kevinluo/Desktop/zillow_data/properties_2016.csv', low_memory=False)
# field is named differently in submission
test_df['parcelid'] = test_df['ParcelId']

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
print("Train: ", train_df.shape)
print("Test: ", test_df.shape)

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
y_train = train_df.logerror

for col in x_train.columns:
    if x_train[col].dtype == object:
        print("found")
        print(col)

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

test_df['transactiondate'] = pd.Timestamp('2016-12-01')  # Dummy
test_df = add_date_features(test_df)
X_test = test_df[train_features]

for c in X_test.dtypes[X_test.dtypes == object].index.values:
    X_test[c] = (X_test[c] == True)

"""
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=27000)
print("train/val shapes: ")
print(x_train.shape)
print(x_val.shape)  """

#PARAM SETUP

print('Training ...')

y_mean = y_train.mean()
#XGBoost params
xgb_params = {
    'n_estimators': 500,
    'eta': 0.035,
    'max_depth': 6,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'colsample_bytree': 0.9,
    'reg_lambda': 0.8,
    'reg_alpha': 0.4,
    'base_score': y_mean,
    'eval_metric': 'mae',
    'random_state': 1,
    'seed': 143,
    'silent': True
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
params_1['silent'] = 1
params_1['feature_fraction_seed'] = 2
params_1['bagging_seed'] = 3
params_1['n_estimators'] = 430

#CatBoost params
catboost_params = {
    'iterations': 200,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'loss_function': 'MAE',
    'eval_metric': 'MAE',
    'verbose': False,
}

# rf params
rf_params = {
    'n_estimators': 100,
    'max_depth': 8,
    'min_samples_split': 100,
    'min_samples_leaf': 30,
}

#Extra Trees params
et_params = {
    'n_estimators': 500,
    'max_depth': 8,
    'min_samples_leaf': 2,
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 400,
    'learning_rate' : 0.75
}

#Second Layer XGBoost parameters
xgb_params_2 = {'learning_rate': 0.041320682119474678, 'subsample': 0.7544297265725215,
                'reg_lambda': 3.7966350899885803, 'max_depth': 2, 'reg_alpha': 0.13659517451090278, 'n_estimators': 186,
                'objective': 'reg:linear', 'eval_metric': 'mae', 'silent': 1}

"""xgb_params_2 = {
            'n_estimators': 250,
            'eta': 0.037,
            'max_depth': 5,
            'subsample': 0.80,
            'objective': 'reg:linear',
            'eval_metric': 'mae',
            'lambda': 0.8,
            'alpha': 0.4,
            'silent': 1
        } """

"""xgb_mod = XGBRegressor(**xgb_params)
lgbm_mod = LGBMRegressor(**params_1)
cat_mod = CatBoostRegressor(**catboost_params)
#rf_mod = RandomForestRegressor(**rf_params)
#ada_mod = AdaBoostRegressor(**ada_params)

#Iteration1
#base_learners = [('xgb', xgb_mod), ('lightgbm', lgbm_mod), ('catboost', cat_mod), ('randForest', rf_mod),
                #('adaboost', ada_mod)]

base_learners = [('xgb', xgb_mod), ('lightgbm', lgbm_mod), ('catboost', cat_mod)]

meta_learners = [('gb', XGBRegressor(**xgb_params_2)),
                 ('el', ElasticNet())]

#Iteration1
#meta_learners = [('gb', XGBRegressor(**xgb_params_2))]

param_dicts = {'el':
                  {'alpha': uniform(0.5, 15),
                   'l1_ratio': uniform(0, 1),
                   'tol': uniform(0.0003, 0.1),
                   'max_iter': randint(1000, 2000)},
               'gb':
                   {'learning_rate': uniform(0.03, 0.12),
                    'subsample': uniform(0.7, 0.2),
                    'reg_lambda': uniform(1.5, 5),
                    'max_depth': randint(2, 5),
                    'reg_alpha': uniform(0.1, 4),
                    'n_estimators': randint(100, 300),
                    'colsample_bytree': uniform(0.7, 0.3)},
              }


"""

#CREATE ENSEMBLE MODEL FROM BASE MODELS

ensemble = SuperLearner(folds=4)

print("adding baseline models to ensembler")

#ensemble.add([XGBRegressor(**xgb_params, n_estimators=500), LGBMRegressor(**params_1, n_estimators=430),
              #CatBoostRegressor(**catboost_params, random_seed=1),RandomForestRegressor(**rf_params)])

#ensemble.add([XGBRegressor(**xgb_params, n_estimators=500), CatBoostRegressor(**catboost_params, random_seed=1)])
#0.0524151216027

#ensemble.add([XGBRegressor(**xgb_params, n_estimators=500), CatBoostRegressor(**catboost_params, random_seed=1),
              #LGBMRegressor(**params_1, n_estimators=430)])
#0.0532148583306

ensemble.add([XGBRegressor(**xgb_params), LGBMRegressor(**params_1),
              CatBoostRegressor(**catboost_params)])

ensemble.add_meta(XGBRegressor(**xgb_params_2))

#ensemble.add_meta(ElasticNet(alpha=0.57541585223123481, copy_X=True, fit_intercept=True,
      #l1_ratio=0.57540585223123486, max_iter=1000, normalize=True))
#0.0528034247408

#ensemble.add_meta(ElasticNet())

#TRAINING ENSEMBLE

print("training ensembler")
ensemble.fit(x_train, y_train)

#PREDICTING ON ENSEMBLE

print("predicting on ensembler")
preds = ensemble.predict(X_test)

#BUILDING SUBMISSION

print("building prediction submission: ")
sub = pd.read_csv('/Users/kevinluo/Desktop/zillow_data/submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = preds

print('Writing csv ...')
sub.to_csv('kaggle_submission.csv', index=False, float_format='%.4f')


#Validation prediction:

#preds = ensemble.predict(x_val)
#accuracy = mean_absolute_error(y_val, preds)
#print('validation accuracy: ')
#print(accuracy)

#Initialize all of the baseline models
#xgboost_1 = XGBoost(xgb_params, 500)
#lgb_model_1 = LGBM(params_1)
#catboost = CatBoostModel(catboost_params, cat_feature_inds)

#base_models = [xgboost_1, lgb_model_1]
#base_models = [xgboost_1, lgb_model_1, catboost]
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


########################################  Predict on Kaggle data and generate submission file ##########################

#ensemble_model.generateKaggleSubmission(sample, prop, train_columns)"""