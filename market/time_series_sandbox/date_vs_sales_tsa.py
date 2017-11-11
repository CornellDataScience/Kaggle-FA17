import itertools

import pandas as pd
import numpy as np
import sys
from datetime import datetime

df_train = pd.read_csv(
    '../input/train.csv', usecols=[1,4],skiprows=range(1, 101688779),
    converters={'unit_sales': lambda u: float(u) if float(u) > 0 else 0}
)
print(df_train.head())
print(df_train.tail())
#skiprows=range(1, 124035460)
#print(df_train.store_nbr.unique()) # 54 Different stores
df_train['unit_sales'] =  df_train['unit_sales'].apply(pd.np.log1p)

u_dates = df_train.date.unique()
# u_stores = df_train.store_nbr.unique()
# u_items = df_train.item_nbr.unique()
df_train.set_index(["date"], inplace=True)

df_train.loc[:, "unit_sales"].fillna(0, inplace=True)
# Assume missing entries imply no promotion
# df_train.loc[:, "onpromotion"].fillna("False", inplace=True)

# Calculate means
df_train = df_train.groupby(
    ['date']
)['unit_sales'].mean().to_frame('unit_sales')

print(df_train.head())
#
# data = np.array(store_set[1])
# result = None
# arima = ARIMA(data, [7,1,1])
# result = arima.fit(disp=False)
# #print(result.params)
# pred = result.predict(typ='levels')
# x = [i for i in range(600)]
# i=0
