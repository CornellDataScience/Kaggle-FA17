from base_model import SKLearnBase

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
#import xgboost as xgb


def XGBoostBase(SKLearnBase):


    def __init__(**params):
        clf = xgboost



def XGBoost_9_30_2017(SKLearnBase):

    # XG Boost params
    params = {
        'eta': 0.037,
        'max_depth': 5,
        'subsample': 0.60,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'lambda': 0.8,
        'alpha': 0.4,
        'silent': 1
    }

    x_feats = []
    y_feats = []


def ExtraTrees_9_30_2017(SKLearnBase):

    # Extra Trees params
    params = {
        'n_jobs': -1,
        'n_estimators':500,
        'max_depth': 8,
        'min_samples_leaf': 2,
        'verbose': 0
    }

    x_feats = []
    y_feats = []


def RandomForest_9_30_2017(SKLearnBase):

    # rf params
    params = {'n_estimators': 50,
                 'max_depth': 8,
                 'min_samples_split': 100,
                 'min_samples_leaf': 30,
                 }

    x_feats = []
    y_feats = []

def AdaBoost_9_30_2017(SKLearnBase):

    # AdaBoost parameters
    params = {
        'n_estimators': 400,
        'learning_rate' : 0.75
    }

    x_feats = []
    y_feats = []