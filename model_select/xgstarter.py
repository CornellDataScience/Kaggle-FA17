import numpy as np
import pandas as pd
import xgboost as xgb
import gc
from sklearn.grid_search import GridSearchCV

print('Loading data ...')

train = pd.read_csv('./train_2016_v2.csv')
prop = pd.read_csv('./properties_2016.csv')
sample = pd.read_csv('./sample_submission.csv')

print('Binding to float32')

for c, dtype in zip(prop.columns, prop.dtypes):
	if dtype == np.float64:
		prop[c] = prop[c].astype(np.float32)

print('Creating training set ...')

df_train = train.merge(prop, how='left', on='parcelid')

df_train['N-ValueRatio'] = df_train['taxvaluedollarcnt']/df_train['taxamount']
df_train['N-LivingAreaProp'] = df_train['calculatedfinishedsquarefeet']/df_train['lotsizesquarefeet']
df_train['N-ValueProp'] = df_train['structuretaxvaluedollarcnt']/df_train['landtaxvaluedollarcnt']
df_train['N-TaxScore'] = df_train['taxvaluedollarcnt']*df_train['taxamount']

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode', 'storytypeid','taxdelinquencyflag', 'censustractandblock'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 80000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

print('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

del x_train, x_valid; gc.collect()

print('Training ...')

params = {}
params['eta'] = 0.02 # Needs to be tuned using CV
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 4
params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

del d_train, d_valid

print('Building test set ...')

sample['parcelid'] = sample['ParcelId']

prop['N-ValueRatio'] = prop['taxvaluedollarcnt']/prop['taxamount']
prop['N-LivingAreaProp'] = prop['calculatedfinishedsquarefeet']/prop['lotsizesquarefeet']
prop['N-ValueProp'] = prop['structuretaxvaluedollarcnt']/prop['landtaxvaluedollarcnt']
prop['N-TaxScore'] = prop['taxvaluedollarcnt']*prop['taxamount']

df_test = sample.merge(prop, on='parcelid', how='left')

del prop; gc.collect()

x_test = df_test[train_columns]
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

del df_test, sample; gc.collect()

d_test = xgb.DMatrix(x_test)

del x_test; gc.collect()

print('Predicting on test ...')

p_test = clf.predict(d_test)

del d_test; gc.collect()

sub = pd.read_csv('./sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

print('Writing csv ...')
sub.to_csv('xgb_starter3.csv', index=False, float_format='%.4f') # Thanks to @inversion
