from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from models import LinearRegression, XGBoost
#import plotting packages
#import plotly.offline as py
#py.init_notebook_mode(connected=True)
#import plotly.graph_objs as go
import numpy as np
import pandas as pd
import xgboost as xgb

class Ensembler(object):

    """An instance of this class is an ensemble model.  It has several base models, and an independent model
    which acts on the predictions of the base models.
    """

    def __init__(self, base_models, second_layer_model):

        """ base_models is a list of the ML algorithms whose results are ensembled upon.
            second_layer_model is the model we train on the predictions of the base_models.
        """
        self.base_models = base_models
        self.second_layer_model = second_layer_model


    def train(self, x, y):

        """ Trains the ensemble. Uses cross validation during training to prevent data leakage
            into second layer.
        """
        x = np.array(x)
        y = np.array(y)

        kf = KFold(n_splits=4)
        folds = list(kf.split(x, y))

        first_layer_train_predictions = np.zeros((x.shape[0], len(self.base_models)))

        #train first layer
        for i in range(len(self.base_models)):
            print("training baseline model")
            for j, (train_idx, test_idx) in enumerate(folds):
                x_train_fold = x[train_idx]
                y_train_fold = y[train_idx]
                x_holdout_fold = x[test_idx]
                y_holdout_fold = y[test_idx]
                self.base_models[i].train(x_train_fold, y_train_fold)
                first_layer_train_predictions[test_idx, i] = self.base_models[i].predict(x_holdout_fold)

        #train second layer
        print("first layer train predictions: ")
        print(first_layer_train_predictions)
        #self.second_layer_model.train(first_layer_train_predictions, y)
        self.second_layer_model.fit(first_layer_train_predictions, y)

        #Wipe train history with full train
        for i in range(len(self.base_models)):
            print("Final train: ")
            self.base_models[i].train(x, y)

        #we need this value to generate heatmaps
        return first_layer_train_predictions


    def validation_accuracy(self, x_val, y_val):
        # predict on ensembler
        predictions = self.predict(x_val)
        print('predictions: ')
        print(predictions)
        print("real")
        print(y_val)
        accuracy = mean_absolute_error(y_val, predictions)
        print('Accuracy on validation set: ')
        print(accuracy)


    def predict(self, x):
        first_layer_test_predictions = np.zeros((x.shape[0], len(self.base_models)))
        #predict on first layer
        for i in range(len(self.base_models)):
            print("predicting on first layer")
            first_layer_test_predictions[:, i] = self.base_models[i].predict(x)
        #make final predictions on second layer
        second_layer_predictions = self.second_layer_model.predict(first_layer_test_predictions)
        return second_layer_predictions


    def heatmap(self, first_layer_train_predictions):
        print("building dataframe for correlation visual")

        base_predictions_train = pd.DataFrame({#'RandomForest': first_layer_train_predictions[0].ravel(),
                                               'XGBoost1': first_layer_train_predictions[0].ravel(),
                                                'CatBoost': first_layer_train_predictions[1].ravel(),
                                                'LightGBM':  first_layer_train_predictions[2].ravel()
                                               })

        data = [
            go.Heatmap(
                z=base_predictions_train.astype(float).corr().values,
                x=base_predictions_train.columns.values,
                y=base_predictions_train.columns.values,
                colorscale='Portland',
                showscale=True,
                reversescale=True
            )
        ]

        print("building plot")
        py.plot(data, filename='heatmap')


    def generateKaggleSubmission(self, x_test):

        """This method predicts on the test set and generates a file that can be submitted to Kaggle.
        """

        #REMEMBER TO DO CATBOOST DATA CLEANING HERE AS WELL

        print('Predicting on test ...')

        #sample['parcelid'] = sample['ParcelId']
        #df_test = sample.merge(prop, on='parcelid', how='left')

        #x_test = df_test[train_columns]

        #for c in x_test.dtypes[x_test.dtypes == object].index.values:
            #x_test[c] = (x_test[c] == True)

        categorical_cols = ['transaction_month', 'transaction_day', 'transaction_quarter', 'airconditioningtypeid',
                            'buildingqualitytypeid', 'fips', 'heatingorsystemtypeid', 'propertylandusetypeid',
                            'regionidcity',
                            'regionidcounty', 'regionidneighborhood', 'regionidzip', 'yearbuilt']


        test_predictions = self.predict(x_test)

        sub = pd.read_csv('/Users/kevinluo/Desktop/zillow_data/submission.csv')
        for c in sub.columns[sub.columns != 'ParcelId']:
            sub[c] = test_predictions

        print('Writing csv ...')
        sub.to_csv('kaggle_submission.csv', index=False, float_format='%.4f')