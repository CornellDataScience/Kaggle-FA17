from model_super import Learner

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb


class XGBoost(Learner):

    def __init__(self, params):
        #set algorithm to be xgboost
        super().__init__(None)
        self.params = params

    def train(self, x_train, y_train):
        d_train = xgb.DMatrix(x_train, label=y_train)
        watchlist = [(d_train, 'train')]
        xgb_model = xgb.train(self.params, d_train, 800, watchlist, early_stopping_rounds=100, verbose_eval=10)
        self.algorithm = xgb_model

    def predict(self, x_test):
        d_test = xgb.DMatrix(x_test)
        return self.algorithm.predict(d_test)


class ExtraTrees(Learner):

    def __init__(self, params):
        #set algorithm to be extratrees
        super().__init__(ExtraTreesRegressor(**params))


class RandomForest(Learner):

    def __init__(self, params):
        #set algorithm to be randomforest
        super().__init__(RandomForestRegressor(**params))



class AdaBoost(Learner):

    def __init__(self, params):
        #set algorithm to be adaboost
        super().__init__(AdaBoostRegressor(**params))


class DecisionTree(Learner):

    def __init__(self, params):
        #set algorithm to decisiontree
        super().__init__(DecisionTreeRegressor(**params))


class LinearRegressionModel(Learner):

    def __init__(self, params):
        #set algorithm to linear regression
        super().__init__(LinearRegression())