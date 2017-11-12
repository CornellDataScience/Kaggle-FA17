import itertools
import pandas as pd
from pandas import Series
import numpy as np
import sys
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns

df_train = pd.read_csv(
    '../input/train.csv', usecols=[1, 4], dtype={'onpromotion': str},
    converters={'unit_sales': lambda u: float(u) if float(u) > 0 else 0},
    skiprows=range(1, 84035460)
)
print(df_train)
df_train["unit_sales"] = df_train["unit_sales"].apply(np.log1p)
df_train['unit_sales'] = df_train.groupby('date')['unit_sales'].transform('mean')
df_train = df_train.drop_duplicates(subset='date')
print(df_train)

df_train[['date','unit_sales']].to_csv('date_and_sales.csv.gz', index=False, compression='gzip')

AO = Series(df_train)
AO.plot()
# data = np.array(store_set[1])
# result = None
# arima = ARIMA(data, [7,1,1])
# result = arima.fit(disp=False)
# #print(result.params)
# pred = result.predict(typ='levels')
# x = [i for i in range(600)]
# i=0
