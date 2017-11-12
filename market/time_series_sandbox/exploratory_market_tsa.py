import itertools

import pandas as pd
import numpy as np
import sys
from datetime import datetime

df_train = pd.read_csv(
    '../input/train.csv', usecols=[1, 2, 3, 4],
    converters={'unit_sales': lambda u: float(u) if float(u) > 0 else 0}
)
print(df_train.head())
print(df_train.tail())
sys.exit()
#skiprows=range(1, 124035460)
print(df_train.store_nbr.unique()) # 54 Different stores

store_set = {}
for i in range(1, 55):
    store_set[i] = df_train[df_train.store_nbr == i].iloc[:, :]

#print(store_sets)
print("="*9)
print(store_set[1])
for key in store_set:
    dataframe = store_set[key]
    dataframe = dataframe.groupby(
        ['date','item_nbr', 'store_nbr']
    )['unit_sales'].mean().to_frame('unit_sales')
    store_set[key] = dataframe
print(store_set[1])
