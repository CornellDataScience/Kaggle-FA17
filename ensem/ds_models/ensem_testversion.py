from mlens.ensemble import SuperLearner
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy.stats import uniform, randint
from mlens.preprocessing import EnsembleTransformer
from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator
from sklearn.base import BaseEstimator
from model_super import *
from models import *
from ensembler import *
from tqdm import tqdm
from parameters import *
from data_processing import *

#results of iteration 1 (all models):  example.csv for params, 0.052445
#results of iteration 2 (three models):  iteration2.csv for params, 0.052109
#iteration 5 params gives 40th percentile on Kaggle -- best ensembling score yet

# TODO:  Analyze different base models with gridsearch (RandomForest, Adaboost, Neural Networks, DecisionTree, Lasso)

############################################## Custom Catboost Class ###################################################

class MultiCatBoost(BaseEstimator):

    def __init__(self, parameters, cat_feature_inds):
        self.cat_feature_inds = cat_feature_inds
        self.models = []
        self.parameters = parameters
        for i in range(5):
            self.models.append(CatBoostRegressor(**self.parameters, random_seed=i))

    def fit(self, x_train, y_train):
        for i in tqdm(range(5)):
            self.models[i].fit(x_train, y_train, cat_features=self.cat_feature_inds)
        return self

    def predict(self, x_test):
        result = 0.0
        for model in self.models:
            print("predicting on catboost")
            result += model.predict(x_test, verbose=True)
        result /= 5
        return result

############################################## Helper methods ##########################################################

#Performs gridsearch on the "meta-learners" which predict on the first layer predictions
def evaluateSecondLayer(base_learners, x_train, y_train, meta_learners, param_dicts):
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
    table.to_html('iteration5.html')
    table.to_csv('iteration5.csv', index=False, header=False, sep='\t')


#Adds features to the dataset
def add_date_features(df):
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = df["transactiondate"].dt.month
    df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = df["transactiondate"].dt.quarter
    df.drop(["transactiondate"], inplace=True, axis=1)
    return df


########################################### LOADING DATA ##############################################################

train_df = pd.read_csv('/Users/kevinluo/Desktop/zillow_data/train_2016.csv', parse_dates=['transactiondate'],
                       low_memory=False)
test_df = pd.read_csv('/Users/kevinluo/Desktop/zillow_data/submission.csv', low_memory=False)
properties = pd.read_csv('/Users/kevinluo/Desktop/zillow_data/properties_2016.csv', low_memory=False)
# field is named differently in submission
test_df['parcelid'] = test_df['ParcelId']

########################################## PROCESSING DATA ############################################################

train_df = add_date_features(train_df)
train_df = train_df.merge(properties, how='left', on='parcelid')
test_df = test_df.merge(properties, how='left', on='parcelid')

#Identify columns with many missing values and store them into a variable
exclude_missing = missingValueColumns(train_df)

# Identify columns with only one unique value and store them into a variable
exclude_unique = nonUniqueColumns(train_df)

#Identify columns that we will use for training and store them into a variable
train_features = trainingColumns(train_df, exclude_missing, exclude_unique)

#Identify categorical columns
cat_feature_inds = categoricalColumns(train_df, train_features)

# Handle NA values
train_df.fillna(-1, inplace=True)
test_df.fillna(-1, inplace=True)

#Disregard outliers
train_df = train_df[train_df.logerror > -0.4]
train_df = train_df[train_df.logerror < 0.4]

#Initialize training datasets
x_train = train_df[train_features]
y_train = train_df.logerror

#Handle types so training does not throw errors
for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

#Set up test dataset
test_df['transactiondate'] = pd.Timestamp('2016-12-01')  # Dummy
test_df = add_date_features(test_df)
X_test = test_df[train_features]

#Handle types so testing does not throw errors
for c in X_test.dtypes[X_test.dtypes == object].index.values:
    X_test[c] = (X_test[c] == True)

#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=27000)


"""
#GridSearch


xgb_mod = XGBRegressor(**xgb_params)
lgbm_mod = LGBMRegressor(**params_1)
cat_mod = CatBoostRegressor(**catboost_params)


base_learners = [('xgb', xgb_mod), ('lightgbm', lgbm_mod), ('catboost', cat_mod)]

meta_learners = [('gb', XGBRegressor(**xgb_params_2)),
                 ('el', ElasticNet())]

param_dicts = {'el':
                  {'alpha': uniform(0.5, 15),
                   'l1_ratio': uniform(0, 1),
                   'tol': uniform(0.0003, 0.1),
                   'max_iter': randint(1000, 12000)},
               'gb':
                   {'learning_rate': uniform(0.01, 0.12),
                    'subsample': uniform(0.7, 0.3),
                    'reg_lambda': uniform(0.4, 5),
                    'max_depth': randint(2, 6),
                    'reg_alpha': uniform(0.1, 4),
                    'n_estimators': randint(50, 300),
                    'colsample_bytree': uniform(0.7, 0.3)},
              }

#evaluateSecondLayer(base_learners, x_train, y_train, meta_learners, param_dicts)   """


########################################## Create and Train Ensembler ##################################################

ensemble = SuperLearner(folds=4)

print("adding baseline models to ensembler")

ensemble.add([XGBRegressor(**getXGBParams(y_train)), LGBMRegressor(**params_1),
              MultiCatBoost(catboost_params, cat_feature_inds)])

ensemble.add_meta(XGBRegressor(**xgb_params_2))

print("training ensembler")
ensemble.fit(x_train, y_train)

######################################### PREDICTING ON ENSEMBLE #######################################################

print("predicting on ensembler")
preds = ensemble.predict(X_test)

""""#Validation prediction:

preds = ensemble.predict(x_val)
accuracy = mean_absolute_error(y_val, preds)
print('validation accuracy: ')
print(accuracy) """

######################################### BUILDING KAGGLE SUBMISSION ###################################################

print("building prediction submission: ")
sub = pd.read_csv('/Users/kevinluo/Desktop/zillow_data/submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = preds

print('Writing csv ...')
sub.to_csv('kaggle_submission.csv', index=False, float_format='%.4f')