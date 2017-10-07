from model_super import Learner

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
import numpy as np

class XGBoost(Learner):

    def __init__(self, params, num_rounds):
        #set algorithm to be xgboost
        super().__init__(None)
        self.params = params
        self.num_rounds = num_rounds



    def train(self, x_train, y_train):
        d_train = xgb.DMatrix(x_train, label=y_train)
        watchlist = [(d_train, 'train')]
        xgb_model = xgb.train(self.params, d_train, self.num_rounds, watchlist, early_stopping_rounds=50,
                              verbose_eval=10)
        self.algorithm = xgb_model

    def predict(self, x_test):
        x_test = np.array(x_test)
        d_test = xgb.DMatrix(x_test)
        return self.algorithm.predict(d_test)

class LGBM(Learner):

    def __init__(self, parameters):
        super().__init__(None)
        self.params = parameters

    def train(self, x_train, y_train):
        x_train = lgb.Dataset(x_train, label=y_train)
        self.algorithm = lgb.train(self.params, x_train, num_boost_round=430)

    def predict(self, x_test):
        return self.algorithm.predict(x_test)

class CatBoostModel(Learner):

    def __init__(self, parameters):
        super().__init__(CatBoostRegressor(**parameters))


class ExtraTrees(Learner):

    def __init__(self, params):
        #set algorithm to be extratrees
        super().__init__(ExtraTreesRegressor(**params))


class RandomForest(Learner):

    def __init__(self, params):
        #set algorithm to be randomforest
        super().__init__(BaggingRegressor(RandomForestRegressor(**params), max_samples=0.6, max_features=0.9))



class AdaBoost(Learner):

    def __init__(self, params):
        #set algorithm to be adaboost
        super().__init__(BaggingRegressor(AdaBoostRegressor(**params), max_samples=0.6, max_features=0.9))


class DecisionTree(Learner):

    def __init__(self, params):
        #set algorithm to decisiontree
        super().__init__(BaggingRegressor(DecisionTreeRegressor(**params), max_samples=0.6, max_features=0.9))


class LinearRegressionModel(Learner):

    def __init__(self):
        #set algorithm to linear regression
        super().__init__(LinearRegression())

class RidgeModel(Learner):

    def __init__(self):
        super().__init__(Ridge())