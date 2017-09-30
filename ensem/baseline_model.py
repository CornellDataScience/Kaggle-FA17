import numpy as np
import pandas as pd
import xgboost as xgb

#import needed baseline model packages
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

#import plotting packages
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

class Learner(object):

    def __init__(self, algorithm, params=None):
        if params != None:
            self.algorithm = BaggingRegressor(algorithm(**params), max_samples=0.6, max_features=0.9)
        else:
            self.algorithm = BaggingRegressor(algorithm(), max_samples=0.6, max_features=0.9)

    def train(self, x_train, y_train):
        self.algorithm.fit(x_train, y_train)

    def predict(self, x_test):
        return self.algorithm.predict(x_test)

    def fit(self, x, y):
        return self.algorithm.fit(x, y)

    def feature_importances(self, x, y):
        print(self.algorithm.fit(x, y).feature_importances_)

print('Loading data ...')

train = pd.read_csv('train_2016.csv')
prop = pd.read_csv('properties_2016.csv')
sample = pd.read_csv('submission.csv')

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
x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

#Ignore below two lines
x_test = x_train
y_test = y_train

split = 80000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

print('Building DMatrix...')

#create special test set for xgboost [needed because it's a different framework]
d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)


print('Training ...')

#XGBoost params
xgb_params = {
    'eta': 0.037,
    'max_depth': 5,
    'subsample': 0.60,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'lambda': 0.8,
    'alpha': 0.4,
    'silent': 1
}

# rf params
rf_params = {}
rf_params['n_estimators'] = 50
rf_params['max_depth'] = 8
rf_params['min_samples_split'] = 100
rf_params['min_samples_leaf'] = 30

#Extra Trees params
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 400,
    'learning_rate' : 0.75
}

#Initialize all of the baseline models
random_forest = Learner(RandomForestRegressor, rf_params)
extra_trees = Learner(ExtraTreesRegressor)
ada_boost = Learner(AdaBoostRegressor)
decision_tree = Learner(DecisionTreeRegressor)

#Train all of the baseline models
base_models = [random_forest, extra_trees, ada_boost, decision_tree]
for i in range(len(base_models)):
    print("training baseline model")
    base_models[i].train(x_train, y_train)

#Train XGBoost as well (must be handled separately)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(xgb_params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)


print('Building test set ...')

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')

x_test = df_test[train_columns]

print("x test shape: ")
print(x_test.shape)

for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

d_test = xgb.DMatrix(x_test)

print('Predicting on test ...')

print("x test shape: ")
print(x_test.shape)

#Predict on normal baseline models
first_layer_train_predictions = np.empty((x_train.shape[0], 5))
first_layer_test_predictions = np.empty((x_test.shape[0], 5))
for i in range(len(base_models)):

    print("predicting on model")

    first_layer_train_predictions[:, i]= base_models[i].predict(x_train)
    first_layer_test_predictions[:, i] = base_models[i].predict(x_test)


first_layer_train_predictions[:, 4] = clf.predict(d_train)
first_layer_test_predictions[:, 4] = clf.predict(d_test)

print("building dataframe for correlation visual")

base_predictions_train = pd.DataFrame( {'RandomForest': first_layer_train_predictions[0].ravel(),
    'ExtraTrees': first_layer_train_predictions[1].ravel(),
     'AdaBoost': first_layer_train_predictions[2].ravel(),
      'GradientBoost': first_layer_train_predictions[3].ravel(),
        'XGBoost': first_layer_train_predictions[4].ravel()
    })

print("building data for correlation visual")

data = [
    go.Heatmap(
        z= base_predictions_train.astype(float).corr().values ,
        x=base_predictions_train.columns.values,
        y= base_predictions_train.columns.values,
          colorscale='Portland',
            showscale=True,
            reversescale = True
    )
]

print("building plot")

py.plot(data, filename='labelled-heatmap')

print("Training second layer")

linear_regression = Learner(LinearRegression)
linear_regression.train(first_layer_train_predictions, y_train)

print("Predicting second layer")

second_layer_predictions = linear_regression.predict(first_layer_test_predictions)

sub = pd.read_csv('submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = second_layer_predictions

print('Writing csv ...')
sub.to_csv('submission_file.csv', index=False, float_format='%.4f')