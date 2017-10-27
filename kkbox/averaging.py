from multiprocessing import Pool, cpu_count
import gc; gc.enable()
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import *
import sklearn
import pickle

df1 = pd.read_csv('./submission3.csv')
df2 = pd.read_csv('./submission4.csv')

df2 = pd.merge(df1, df2, on=['msno'], how='inner')
df2['is_churn'] = (df2['is_churn_x'] + df2['is_churn_y']) / 2.0

# df2['is_churn'] = df2['is_churn'].apply(lambda x: x*0.5)
df2[['msno','is_churn']].to_csv('submission5.csv.gz', index=False, compression='gzip')
