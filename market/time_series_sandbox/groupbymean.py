import pandas as pd
import numpy as np
import sys
from datetime import datetime

df_train = pd.read_csv(
    './groupbytest.csv',
    converters={'count': lambda u: float(u) if float(u) > 0 else 0}
)

u_dates = df_train.date.unique()
u_stores = df_train.store.unique()
u_items = df_train.item.unique()

print(u_dates)
print(u_stores)
print(u_items)
print("\n")
df_train.set_index(["date", "store", "item"], inplace=True)

print(df_train)


df_train = df_train.groupby(
    ['date','store', 'item']
)['count'].mean().to_frame('unit_sales')
print("AFTER GROUP")
print(df_train)
print(df_train[2:3])
