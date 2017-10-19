import math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import pickle

clf = LinearSVR()

train = pd.read_csv('./train_2016_v2.csv')
prop = pd.read_csv('./properties_2016.csv')
sample = pd.read_csv('./sample_submission.csv')

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print('Binding to float32')

for c, dtype in zip(prop.columns, prop.dtypes):
	if dtype == np.float64:
		prop[c] = prop[c].astype(np.float32)

print('Creating training set ...')

df_train = train.merge(prop, how='left', on='parcelid')
df_train.convert_objects(convert_numeric=True)
df_train.fillna(value=0, inplace=True)
X = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode', 'storytypeid', 'hashottuborspa', 'fireplaceflag','garagetotalsqft', 'pooltypeid7', 'propertylandusetypeid', 'taxdelinquencyflag', 'censustractandblock'], axis=1)
X = preprocessing.normalize(X)
Y = df_train['logerror'].values

weird = (X.applymap(type) != X.iloc[0].apply(type)).any(axis=1)
new_df = X[weird]
new_df.to_csv('weird.csv')
print('done writing weird to csv')


# clf = LinearRegression(n_jobs=-1)
# clf.fit(X_train, y_train)
# confidence = clf.score(X_test, y_test)
print('About to divide into cross validation')
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

clf = LinearSVR()
clf.fit(X_train, Y_train)
confidence = clf.score(X_test, Y_test)
print(confidence)
filename = 'zillow_linear_svr.pickle'
with open(filename,'wb') as f:
    pickle.dump(clf, f)
