import itertools

import pandas as pd
import numpy as np
import datetime

count = 0

df = pd.read_pickle("./train_pickle.pkl")
del df["store_nbr"]
del df["item_nbr"]
del df["onpromotion"]
df2 = df[:120000]
del df
print(df2.tail(10))


def create_weekday(row):
    res = datetime.datetime(int(row['Year']), int(row['Month']), int(row['Day'])).weekday()
    print(res)
    return res

df2['weekday'] = df2.apply(lambda row: create_weekday(row), axis=1)

def cust_mean(grp):
    grp['mean'] = grp['unit_sales'].mean()
    print("calculated mean")
    return grp

df2 = df2.groupby(['weekday']).apply(cust_mean)

df2[['unit_sales','weekday']].to_csv('weekday_sales4.csv.gz', index=False, compression='gzip')

print(df2.tail(10))
print(df2.dtypes)
