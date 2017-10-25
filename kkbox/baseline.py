from multiprocessing import Pool, cpu_count
import gc; gc.enable()
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import *
import sklearn
import pickle

# train = pd.read_csv('./train.csv')
# test = pd.read_csv('./sample_submission_zero.csv')
#
# transactions = pd.read_csv('./transactions.csv', usecols=['msno'])
# transactions = pd.DataFrame(transactions['msno'].value_counts().reset_index())
# transactions.columns = ['msno','trans_count']
# train = pd.merge(train, transactions, how='left', on='msno')
# test = pd.merge(test, transactions, how='left', on='msno')
# transactions = []; print('transaction merge...')

# user_logs = pd.read_csv('../input/user_logs.csv', usecols=['msno'])
# user_logs = pd.DataFrame(user_logs['msno'].value_counts().reset_index())
# user_logs.columns = ['msno','logs_count']
# train = pd.merge(train, user_logs, how='left', on='msno')
# test = pd.merge(test, user_logs, how='left', on='msno')
# user_logs = []; print('user logs merge...')
#
# members = pd.read_csv('./members.csv')
# train = pd.merge(train, members, how='left', on='msno')
# test = pd.merge(test, members, how='left', on='msno')
# members = []; print('members merge...')
#
#
# transactions = pd.read_csv('./transactions.csv')
# transactions = transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)
# transactions = transactions.drop_duplicates(subset=['msno'], keep='first')
#
# train = pd.merge(train, transactions, how='left', on='msno')
# test = pd.merge(test, transactions, how='left', on='msno')
#
# #Encode male as 1 and female as 2
# gender = {'male':1, 'female':2}
# train['gender'] = train['gender'].map(gender)
# test['gender'] = test['gender'].map(gender)
#
# train = train.fillna(0)
# test = test.fillna(0)
#
# transactions=[]
# def transform_df(df):
#     df = pd.DataFrame(df)
#     df = df.sort_values(by=['date'], ascending=[False])
#     df = df.reset_index(drop=True)
#     df = df.drop_duplicates(subset=['msno'], keep='first')
#     return df
#
# def transform_df2(df):
#     df = df.sort_values(by=['date'], ascending=[False])
#     df = df.reset_index(drop=True)
#     df = df.drop_duplicates(subset=['msno'], keep='first')
#     return df
# train = train.fillna(0)
# test = test.fillna(0)
#
# cols = [c for c in train.columns if c not in ['is_churn','msno']]

# train.to_pickle("combined_train.pkl")
# test.to_pickle("combined_test.pkl")
train = pd.read_pickle("combined_train.pkl")
test = pd.read_pickle("combined_test.pkl")

cols = [c for c in train.columns if c not in ['is_churn','msno']]

def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', metrics.log_loss(labels, preds)

fold = 1
for i in range(fold):
    params = {
        'eta': 0.02, #use 0.002
        'max_depth': 7,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': i,
        'silent': True
    }
    x1, x2, y1, y2 = model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.3, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 150,  watchlist, feval=xgb_score, maximize=False, verbose_eval=50, early_stopping_rounds=50) #use 1500
    if i != 0:
        pred += model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
    else:
        pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
pred /= fold
test['is_churn'] = pred.clip(0.0000001, 0.999999)
test[['msno','is_churn']].to_csv('submission3.csv.gz', index=False, compression='gzip')
