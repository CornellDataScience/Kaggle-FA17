import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from tqdm import tqdm
from models import *
from ensembler import *
from parameters import *

#Best validation score: 0.0518398510732 (Ridge Regression as decider, LGBM and XGBoost, one each)
#Other scores:  0.0519503944372 (Dropped DecisionTree and RandomForest)

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

#identify columns with many missing values
exclude_missing = missingValueColumns(train_df)

# exclude where we only have one unique value
exclude_unique = nonUniqueColumns(train_df)

# identify training features
train_features = trainingColumns(train_df, exclude_missing, exclude_unique)

#identify categorical columns
cat_feature_inds = categoricalColumns(train_df, train_features)

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

#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=27000)
#print("validation shape: ")
#print(y_val.shape)

test_df['transactiondate'] = pd.Timestamp('2016-12-01')  # Dummy
test_df = add_date_features(test_df)
X_test = test_df[train_features]
for c in X_test.dtypes[X_test.dtypes == object].index.values:
    X_test[c] = (X_test[c] == True)
print(X_test.shape)

######################################################## Training ######################################################

print('Training ...')

#Initialize all of the baseline models
xgboost_1 = XGBoost(xgb_params, 500)
lgb_model_1 = LGBM(params_1)
catboost = CatBoostModel(catboost_params, cat_feature_inds)
#randomForest = RandomForest(rf_params)

base_models = [xgboost_1, catboost, lgb_model_1]
#create an ensembler with the base_models
ensemble_model = Ensembler(base_models, XGBRegressor(**xgb_params_2))

#train ensembler and save necessary values to create a heatmap
first_layer_results = ensemble_model.train(x_train, y_train)
#create heatmap
ensemble_model.heatmap(first_layer_results)

############################################# Evaluate on validation set ###############################################

#ensemble_model.validation_accuracy(x_val, y_val)

########################################  Predict on Kaggle data and generate submission file ##########################

ensemble_model.generateKaggleSubmission(X_test)