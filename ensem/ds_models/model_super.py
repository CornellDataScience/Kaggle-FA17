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


    def fit_with_analytics(bundle=None):

        """ This is fit but also returns information for the roc curve
        """
        
        if bundle is None:
            bundle = self.fit()

        prediction = self.predict(raw_bundle['X_test'])
        

        return {'model': self, 
                'X_train': X_train,
                'y_train': y_train,
                'analytics': self.get_analytics(raw_bundle['y_test'], prediction)
                 }

    def get_analytics(self, y_true, y_predicted):

        """ Returns a dictionary of multiple scores
        """

        return {'roc_auc_curve': roc_auc_score(y_true, y_predicted),
                'roc_curve': roc_curve(y_true, y_predicted),
                'accuracy_score': accuracy_score(y_true, y_predicted),
                }


    def feature_importances(self, x, y):

        """ A wrapper for the feature importances
        """
        return self.clf.feature_importances_