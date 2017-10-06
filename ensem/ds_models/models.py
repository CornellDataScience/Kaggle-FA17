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
        xgb_model = xgb.train(self.params, d_train, 500, watchlist, early_stopping_rounds=50, verbose_eval=10)
        self.algorithm = xgb_model

    def predict(self, x_test):
        d_test = xgb.DMatrix(x_test)
        return self.algorithm.predict(d_test)


class ExtraTrees(Learner):

    def __init__(self, params):
        #set algorithm to be extratrees
        super().__init__(BaggingRegressor(ExtraTreesRegressor(**params), max_samples=0.6, max_features=0.9))


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

    def __init__(self, params):
        #set algorithm to linear regression
        super().__init__(LinearRegression())