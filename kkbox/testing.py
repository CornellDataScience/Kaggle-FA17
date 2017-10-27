from multiprocessing import Pool, cpu_count
import gc; gc.enable()
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import *
import sklearn
import pickle

df_transactions = pd.read_csv('./transactions.csv')
train = pd.read_csv('./train.csv')
test = pd.read_csv('./sample_submission_zero.csv')

train = pd.merge(train, df_transactions, how='left', on='msno')
test = pd.merge(test, df_transactions, how='left', on='msno')

del train['transaction_date']
del train['membership_expire_date']


del test['transaction_date']
del test['membership_expire_date']



print("Successfully deleted columns")
