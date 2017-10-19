import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb
import datetime as dt
import gc

print('loading files...')
prop = pd.read_csv('properties_2016.csv',low_memory=False)
prop.rename(columns={'parcelid': 'ParcelId'}, inplace=True)   # make it the same as sample_submission
train = pd.read_csv('train_2016_v2.csv')
train.rename(columns={'parcelid': 'ParcelId'},inplace=True)
sample = pd.read_csv('sample_submission.csv')
print(train.shape, prop.shape, sample.shape)

print('preprocessing, fillna, outliters, dtypes ...')

prop['longitude']=prop['longitude'].fillna(prop['longitude'].median()) / 1e6   #  convert to float32 later
prop['latitude'].fillna(prop['latitude'].median()) / 1e6
prop['censustractandblock'].fillna(prop['censustractandblock'].median()) / 1e12
train = train[train['logerror'] <  train['logerror'].quantile(0.9975)]  # exclude 0.5% of outliers
train = train[train['logerror'] >  train['logerror'].quantile(0.0025)]

print('qualitative ...')
qualitative = [f for f in prop.columns if prop.dtypes[f] == object]
prop[qualitative] = prop[qualitative].fillna('Missing')
for c in qualitative:  prop[c] = LabelEncoder().fit(list(prop[c].values)).transform(list(prop[c].values)).astype(int)

print('smallval ...')
smallval = [f for f in prop.columns if np.abs(prop[f].max())<100]
prop[smallval] = prop[smallval].fillna('Missing')
for c in smallval:  prop[c] = LabelEncoder().fit(list(prop[c].values)).transform(list(prop[c].values)).astype(np.int8)

print('other ...')
other=['regionidcounty','fips','propertycountylandusecode','propertyzoningdesc','propertylandusetypeid']
prop[other] = prop[other].fillna('Missing')
for c in other:  prop[c] = LabelEncoder().fit(list(prop[c].values)).transform(list(prop[c].values)).astype(int)

randomyears=pd.Series(np.random.choice(prop['yearbuilt'].dropna().values,len(prop)))
prop['yearbuilt']=prop['yearbuilt'].fillna(randomyears).astype(int)
med_yr=prop['yearbuilt'].quantile(0.5)
prop['New']=prop['yearbuilt'].apply(lambda x: 1 if x > med_yr else 0).astype(np.int8)  # adding a new feature

randomyears=pd.Series(np.random.choice(prop['assessmentyear'].dropna().values,len(prop)))
prop['assessmentyear']=prop['assessmentyear'].fillna(randomyears).astype(int)

prop['unitcnt'] = prop['unitcnt'].fillna(1).astype(int)

feat_to_drop=[ 'finishedsquarefeet50', 'finishedfloor1squarefeet', 'finishedsquarefeet15', 'finishedsquarefeet13']
prop.drop(feat_to_drop,axis=1,inplace=True)   # drop because too many missing values
prop['lotsizesquarefeet'].fillna(prop['lotsizesquarefeet'].quantile(0.001),inplace=True)
prop['finishedsquarefeet12'].fillna(prop['finishedsquarefeet12'].quantile(0.001),inplace=True)
prop['calculatedfinishedsquarefeet'].fillna(prop['finishedsquarefeet12'],inplace=True)
prop['taxamount'].fillna(prop['taxamount'].quantile(0.001),inplace=True)
prop['landtaxvaluedollarcnt'].fillna(prop['landtaxvaluedollarcnt'].quantile(0.001),inplace=True)
prop.fillna(0,inplace=True)

print('quantitative ...')
quantitative = [f for f in prop.columns if prop.dtypes[f] == np.float64]
prop[quantitative] = prop[quantitative].astype(np.float32)

cfeatures = list(prop.select_dtypes(include = ['int64', 'int32', 'uint8', 'int8']).columns)
for c in qualitative:  prop[c] = LabelEncoder().fit(list(prop[c].values)).transform(list(prop[c].values))

# some quantitative features have a limited number of values (eg ZIP code)
for c in ['rawcensustractandblock',  'regionidcity',  'regionidneighborhood',  'regionidzip',  'censustractandblock'] :
    prop[c] = LabelEncoder().fit(list(prop[c].values)).transform(list(prop[c].values))

# other quantitative features were probably transformed when Zillow first calculate prices because of the skew
for c in ['calculatedfinishedsquarefeet', 'finishedsquarefeet12', 'lotsizesquarefeet',
    'structuretaxvaluedollarcnt',  'taxvaluedollarcnt',  'landtaxvaluedollarcnt',  'taxamount'] :
    prop[c] = np.log1p(prop[c].values)

gc.collect()

print('create new features and the final dataframes frames ...')

#replace latitudes and longitudes with 500 clusters  (similar to ZIP codes)
coords = np.vstack(prop[['latitude', 'longitude']].values)
sample_ind = np.random.permutation(len(coords))[:1000000]
kmeans = MiniBatchKMeans(n_clusters=500, batch_size=100000).fit(coords[sample_ind])
prop['Cluster'] = kmeans.predict(prop[['latitude', 'longitude']])

prop['Living_area_prop'] = prop['calculatedfinishedsquarefeet'] / prop['lotsizesquarefeet']
prop['Value_ratio'] = prop['taxvaluedollarcnt'] / prop['taxamount']
prop['Value_prop'] = prop['structuretaxvaluedollarcnt'] / prop['landtaxvaluedollarcnt']
prop['Taxpersqrtfoot']=prop['finishedsquarefeet12']/prop['taxamount']

train['transactiondate'] = pd.to_datetime(train.transactiondate)
train['Month'] = train['transactiondate'].dt.month.astype(np.int8)
train['Day'] = train['transactiondate'].dt.day.astype(np.int8)
train['Season'] = train['Month'].apply(lambda x: 1 if x in [1,2,9,10,11,12] else 0).astype(np.int8)

month_err=(train.groupby('Month').aggregate({'logerror': lambda x: np.mean(x)})- train['logerror'].mean()).values
train['Meanerror']=train['Month'].apply(lambda x: month_err[x-1]).astype(np.float)

train['abserror']=train['logerror'].abs()
month_abs_err=(train.groupby('Month').aggregate({'abserror': lambda x: np.mean(x)})- train['abserror'].mean()).values
train['Meanabserror']=train['Month'].apply(lambda x: month_abs_err[x-1]).astype(np.float)
train.drop(['abserror'], axis=1,inplace=True)

X = train.merge(prop, how='left', on='ParcelId')
y = X['logerror']
X.drop(['ParcelId', 'logerror', 'transactiondate'], axis=1,inplace=True)
features=list(X.columns)

print(X.shape, y.shape)
gc.collect()

print('training xgboost ...')
X_train=X
y_train=y
y_mean = np.mean(y_train)
xgb_params = {'eta': 0.037, 'max_depth': 5, 'subsample': 0.80,  'eval_metric': 'mae',
              'lambda': 0.8,   'alpha': 0.4, 'base_score': y_mean, 'silent': 1 }
dtrain = xgb.DMatrix(X_train, y_train)
model = xgb.train(xgb_params, dtrain, num_boost_round=250)
pred1 = model.predict(dtrain)
print(' xgb MAE train  {:.4f}'.format(np.mean(np.abs(y_train.values-pred1) )))
del dtrain, pred1
gc.collect()

fig, ax = plt.subplots(figsize=(20, 20))
xgb.plot_importance(model, ax=ax)
plt.show()
gc.collect()

print('dropping insignificant features...')
X.drop(['pooltypeid10','buildingclasstypeid','fullbathcnt','pooltypeid7','poolsizesum','threequarterbathnbr',
'decktypeid','yardbuildingsqft17','airconditioningtypeid', 'hashottuborspa', 'pooltypeid2', 'basementsqft',
'fireplacecnt','heatingorsystemtypeid','roomcnt','yardbuildingsqft26'], axis=1, inplace=True)

print('building new DMatrix...')
X_train_new = X
x_test = prop.drop(['Parcelid'], axis=1)
dtrain_new = xgb.DMatrix(X_train_new, y_train)
dtest = xgb.DMatrix(x_test)
xgb_params_test = {
    'eta': 0.033,
    'max_depth': 6,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 1
}

print('calculating cross validation...')
cv_result = xgb.cv(xgb_params_test,
                   dtrain_new,
                   nfold=5,
                   num_boost_round=500,
                   early_stopping_rounds=5,
                   verbose_eval=10,
                   show_stdv=False
                  )
num_boost_rounds = len(cv_result)
print(num_boost_rounds)
# train model

print('training new model...')
model_test = xgb.train(dict(xgb_params_test, silent=1), dtrain_new, num_boost_round=num_boost_rounds)
pred_test = model_test.predict(dtest)
y_pred=[]


for i,predict in enumerate(pred_test):
    y_pred.append(str(round(predict,4)))
y_pred=np.array(y_pred)

print('writinf to csv...')
output = pd.DataFrame({'ParcelId': prop['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})
# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
from datetime import datetime
output.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
