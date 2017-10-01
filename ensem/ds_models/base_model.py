from __future__ import division


import pandas as pd
import ds_models.utils as dsu
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, accuracy_score
from features import *



def SKLearnBase(object):

    def __init__():

        """ This MUST be overwritten
        """
        raise NotImplementedErrror('subclass must provide their own init')



    def pull_and_create_df(self):

        """ This will pull the data and create a df from it.
        """

        all_feats = self.x_feats, self.y_feats
        raw_df =  dsu.create_df(all_feats)

        return {'raw_df': raw_df,
                'raw_X_df': raw_df[train_features],
                'raw_y_df': raw_df[test_features]
                }


    def test_train_split(self, raw_X, raw_y, test_size=0.8):

        """ This will split the data into a test and train split.
        """
        
        X_train, X_test, y_train, y_test = train_test_split(
            raw_X, 
            raw_y, 
            test_size=test_size, 
            random_state=seed)
        
        return {'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'raw_X': raw_X,
                'raw_y': raw_y
                }

    def train(self, X_train, y_train):

        """ Just a wrapper for training the data
        """
        self.clf.fit(x_train, y_train)


    def fit(self, bundle=None):

        """ This is training and also returning a whole bundle to work with.
        """

        if bundle is None:
            
            raw_bundle = self.pull_and_create_df()
            bundle = self.test_train_split(raw_bundle['X_raw'], raw_bundle['y_raw'])
        
        self.train(X_train, y_train)
        
        return {'model': self, 
                'X_train': X_train,
                'y_train': y_train,
                'X_test': bundle['y_test'],
                'y_test': bundle['X_test']
                 }


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



    def predict(self, X_test):

        """ A wrapper for the prediction method 
        """

        return self.clf.predict(X_test)


    def feature_importances(self, x, y):

        """ A wrapper for the feature importances
        """
        return self.clf.feature_importances_

def EnsembleBase(object):

    def __init__(self, classifiers, x_feats, y_feats):

        """ classifiers is a list of lists. The zeroeth list
            is the list of first classifiers. The next list builds
            a prediction based on the predictions of the previous
            classifiers on the list.
            x_feats: features that will be trained on 
            y_feats: features that will be used predictions
        """

        self.classifiers = classifiers


    def train(self):

        """ Trains the classifiers 
        """

        clf_predictions = pd.DataFrame()

        for clf in classifiers[0]:
            bundle = clf.fit()
            col = clf.predict(bundle['X_train'])
            col = col.append(clf.predict(bundle['X_test']))
            clf_predictions[clf.name] = col

        for stack in classifiers[1:]:
            for clf in stack:
                clf_predictions


    def get_oof(clf, x_train, y_train, x_test):

        """ This code was taken from Anisotrophic where he
            described and implemented this cross-validation technique
        """

        ntrain = train.shape[0]
        ntest = test.shape[0]

        NFOLDS = 5 # set folds for out-of-fold prediction
        kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


    def fit(self):
        asdf

    def correlation(self):
        asfd



