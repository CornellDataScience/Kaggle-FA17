#Baseline untuned XGB kernel

from multiprocessing import Pool, cpu_count
import gc; gc.enable()
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import *
import sklearn
import pickle

train = pd.read_csv('./train.csv')
test = pd.read_csv('./sample_submission_zero.csv')

df_members = pd.read_csv('./members.csv')
df_members = df_members.drop_duplicates(subset=['msno'], keep='first')
df_transactions = pd.read_csv('./transactions.csv')
df_transactions = df_transactions.drop_duplicates(subset=['msno'], keep='first')
df_transactions = df_transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)
def change_datatype(df):
    int_cols = list(df.select_dtypes(include=['int']).columns)
    for col in int_cols:
        if ((np.max(df[col]) <= 127) and(np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif ((np.max(df[col]) <= 32767) and(np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif ((np.max(df[col]) <= 2147483647) and(np.min(df[col] >= -2147483648))):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)

def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)

# change_datatype(df_transactions)
# change_datatype_float(df_transactions)
# change_datatype(df_members)
# change_datatype_float(df_members)


#Feature Engineering for Transactions
df_transactions['discount'] = df_transactions['plan_list_price'] - df_transactions['actual_amount_paid']
df_transactions['is_discount'] = df_transactions.discount.apply(lambda x: 1 if x > 0 else 0)
#df_transactions['amt_per_day'] = df_transactions['actual_amount_paid'] / df_transactions['payment_plan_days']

date_cols = ['transaction_date', 'membership_expire_date']

for col in date_cols:
    df_transactions[col] = pd.to_datetime(df_transactions[col], format='%Y%m%d')

df_transactions['membership_duration'] = df_transactions.membership_expire_date - df_transactions.transaction_date
df_transactions['membership_duration'] = df_transactions['membership_duration'] / np.timedelta64(1, 'D')
df_transactions['membership_duration'] = df_transactions['membership_duration'].astype(int)
df_transactions['transaction_date'] = df_transactions['transaction_date'].astype(int)
#df_transactions['membership_expire_date'] = df_transactions['membership_expire_date'].astype(int)

# Feature Engineering for Members
df_members = pd.read_csv('./members.csv')
date_cols = ['registration_init_time', 'expiration_date']
for col in date_cols:
    df_members[col] = pd.to_datetime(df_members[col], format='%Y%m%d')
df_members['registration_duration'] = df_members.expiration_date - df_members.registration_init_time
df_members['registration_duration'] = df_members['registration_duration'] / np.timedelta64(1, 'D')
df_members['registration_duration'] = df_members['registration_duration'].astype(int)
df_members['registration_init_time'] = df_members['registration_init_time'].astype(int)
#df_members['expiration_date'] = df_members['expiration_date'].astype(int)

df_comb = pd.merge(df_transactions, df_members, on='msno', how='inner')

#--- deleting the dataframes to save memory
del df_transactions
del df_members

df_comb['reg_mem_duration'] = df_comb['registration_duration'] - df_comb['membership_duration']
#df_comb['autorenew_&_not_cancel'] = ((df_comb.is_auto_renew == 1) == (df_comb.is_cancel == 0)).astype(np.int8)
# df_comb['notAutorenew_&_cancel'] = ((df_comb.is_auto_renew == 0) == (df_comb.is_cancel == 1)).astype(np.int8)
df_comb['long_time_user'] = (((df_comb['registration_duration'] / 365).astype(int)) > 1).astype(int)

train = pd.merge(train, df_comb, how='left', on='msno')
test = pd.merge(test, df_comb, how='left', on='msno')


#Encode male as 1 and female as 2
gender = {'male':1, 'female':2}
train['gender'] = train['gender'].map(gender)
test['gender'] = test['gender'].map(gender)
train.replace([np.inf, -np.inf], np.nan)
test.replace([np.inf, -np.inf], np.nan)
train = train.fillna(0)
test = test.fillna(0)


def transform_df(df):
    df = pd.DataFrame(df)
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

def transform_df2(df):
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

del train['transaction_date']
del train['membership_expire_date']
del train['registration_init_time']
del train['expiration_date']
del test['transaction_date']
del test['membership_expire_date']
del test['registration_init_time']
del test['expiration_date']

print(train.head())

cols = [c for c in train.columns if c not in ['is_churn','msno']]


# train = pd.read_pickle("combined_train.pkl")
# test = pd.read_pickle("combined_test.pkl")

# Feature Engineering


def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', metrics.log_loss(labels, preds)

fold = 1
for i in range(fold):
    params = {
        'eta': 0.002, #use 0.002
        'max_depth': 7,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': i,
        'silent': True
    }
    x1, x2, y1, y2 = model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.3, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 1600,  watchlist, feval=xgb_score, maximize=False, verbose_eval=1, early_stopping_rounds=1500) #use 1500
    if i != 0:
        pred += model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
    else:
        pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
pred /= fold
test['is_churn'] = pred.clip(0.0000001, 0.999999)
test[['msno','is_churn']].to_csv('submission3.csv.gz', index=False, compression='gzip')

train.to_pickle("combined_train.pkl")
test.to_pickle("combined_test.pkl")
