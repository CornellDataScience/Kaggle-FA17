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
        self.second_layer_model.train(first_layer_train_predictions, y_val)

        #we need this value to generate heatmaps
        return first_layer_train_predictions


    def validation_accuracy(self, x_val, y_val):
        # predict on ensembler
        predictions = self.predict(x_val)
        accuracy = mean_absolute_error(y_val, predictions)
        print('Accuracy on validation set: ')
        print(accuracy)


    def predict(self, x):
        first_layer_test_predictions = np.zeros((x.shape[0], 5))
        #predict on first layer
        for i in range(len(self.base_models)):
            print("predicting on first layer")
            first_layer_test_predictions[:, i] = self.base_models[i].predict(x)
        #make final predictions on second layer
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
        py.plot(data, filename='heatmap')


    def generateKaggleSubmission(self, sample, prop, train_columns):

        """This method predicts on the test set and generates a file that can be submitted to Kaggle.
        """

        print('Building test set ...')

        sample['parcelid'] = sample['ParcelId']
        df_test = sample.merge(prop, on='parcelid', how='left')

        x_test = df_test[train_columns]

        for c in x_test.dtypes[x_test.dtypes == object].index.values:
            x_test[c] = (x_test[c] == True)

        categorical_cols = ['transaction_month', 'transaction_day', 'transaction_quarter', 'airconditioningtypeid',
                            'buildingqualitytypeid', 'fips', 'heatingorsystemtypeid', 'propertylandusetypeid',
                            'regionidcity',
                            'regionidcounty', 'regionidneighborhood', 'regionidzip', 'yearbuilt']
        #mean normalization
        for column in x_test:
            if column not in categorical_cols:
                mean = x_test[column].mean()
                stdev = x_test[column].std()
                if stdev != 0:
                    x_test[column] = (x_test[column] - mean) / stdev

        test_predictions = self.predict(x_test)

        sub = pd.read_csv('/Users/kevinluo/Desktop/zillow_data/submission.csv')
        for c in sub.columns[sub.columns != 'ParcelId']:
            sub[c] = test_predictions

        print('Writing csv ...')
        sub.to_csv('kaggle_submission.csv', index=False, float_format='%.4f')