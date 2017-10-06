from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from models import LinearRegression, XGBoost
#import plotting packages
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
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
        #Split into test and validation set - we want to send validation results to next layer
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=18000)
        first_layer_train_predictions = np.zeros((18000, 5))

        #train first layer
        for i in range(len(self.base_models)):
            print("training baseline model")
            self.base_models[i].train(x_train, y_train)
            first_layer_train_predictions[:, i] = self.base_models[i].predict(x_val)

        #train second layer
        print("second layer dataset: ")
        print(first_layer_train_predictions)
        print("real values: ")
        print(y_val)
        #self.second_layer_model.train(first_layer_train_predictions, y_val)
        self.second_layer_model.train(first_layer_train_predictions, y_val)

        #we need this value to generate heatmaps
        return first_layer_train_predictions


    def predict(self, x, y):
        first_layer_test_predictions = np.zeros((x.shape[0], 5))
        #predict on first layer
        for i in range(len(self.base_models)):
            print("predicting on first layer")
            first_layer_test_predictions[:, i] = self.base_models[i].predict(x)
        #make final predictions on second layer
        print("what the second layer features look like: ")
        print(first_layer_test_predictions)
        print("features shape: ")
        print(first_layer_test_predictions.shape)
        print('labels shape: ')
        print(y.shape)
        second_layer_predictions = self.second_layer_model.predict(first_layer_test_predictions)
        return second_layer_predictions


    def heatmap(self, first_layer_train_predictions):
        print("building dataframe for correlation visual")

        base_predictions_train = pd.DataFrame({'RandomForest': first_layer_train_predictions[0].ravel(),
                                               'ExtraTrees': first_layer_train_predictions[1].ravel(),
                                               'AdaBoost': first_layer_train_predictions[2].ravel(),
                                               'GradientBoost': first_layer_train_predictions[3].ravel(),
                                               'XGBoost': first_layer_train_predictions[4].ravel()
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
        py.plot(data, filename='labelled-heatmap')


    def generateKaggleSubmission(self, sample, prop, train_columns):

        """This method predicts on the test set and generates a file that can be submitted to Kaggle.
        This method is only partially completed.
        """

        print('Building test set ...')

        sample['parcelid'] = sample['ParcelId']
        df_test = sample.merge(prop, on='parcelid', how='left')

        x_test = df_test[train_columns]

        for c in x_test.dtypes[x_test.dtypes == object].index.values:
            x_test[c] = (x_test[c] == True)

        test_predictions = self.predict(x_test)

        sub = pd.read_csv('submission.csv')
        for c in sub.columns[sub.columns != 'ParcelId']:
            sub[c] = test_predictions

        print('Writing csv ...')
        sub.to_csv('submission_file.csv', index=False, float_format='%.4f')