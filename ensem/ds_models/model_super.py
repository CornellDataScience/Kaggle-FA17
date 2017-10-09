from __future__ import division


import pandas as pd
#import ds_models.utils as dsu
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, accuracy_score
#from features import *


class Learner(object):

    def __init__(self, algorithm):

        """ This MUST be overwritten
        """
        self.algorithm = algorithm


    def train(self, x_train, y_train):

        """ Just a wrapper for training the data
        """
        self.algorithm.fit(x_train, y_train)


    def predict(self, x_test):

        """ A wrapper for the prediction method
        """

        return self.algorithm.predict(x_test)


    def feature_importances(self, x, y):

        """ A wrapper for the feature importances
        """
        return self.clf.feature_importances_