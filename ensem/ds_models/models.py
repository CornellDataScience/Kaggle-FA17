from model_super import Learner

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb


class XGBoost(Learner):

    def __init__(self, params):
        #set algorithm to be xgboost
        super(xgb, params)

    def train(self, x_train, y_train):
        d_train = xgb.DMatrix(x_train, label=y_train)
        xgb_model = self.algorithm.train(xgb_params, d_train, early_stopping_rounds=100, verbose_eval=10)
        self.algorithm = xgb_model

    def predict(self, x_test):
        d_test = xgb.DMatrix(x_test)
        return self.algorithm.predict(d_test)


class ExtraTrees(Learner):

    def __init__(self, params):
        #set algorithm to be extratrees
        super(ExtraTreesRegressor, params)


class RandomForest(Learner):

    def __init__(self, params):
        #set algorithm to be randomforest
        super(RandomForestRegressor, params)


class AdaBoost(Learner):

    def __init__(self, params):
        #set algorithm to be adaboost
        super(AdaBoostRegressor, params)


class DecisionTree(Learner):

    def __init__(self, params):
        #set algorithm to decisiontree
        super(DecisionTreeRegressor, params)


class LinearRegression(Learner):

    def __init__(self, params):
        #set algorithm to linear regression
        super(LinearRegression, params)