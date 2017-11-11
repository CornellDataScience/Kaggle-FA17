import itertools
import pandas as pd
from pandas import Series
import numpy as np
import sys
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns

df_train = pd.read_csv(
    '../input/train.csv', usecols=[1, 4, ], dtype={'onpromotion': str},
    converters={'unit_sales': lambda u: float(u) if float(u) > 0 else 0},
    skiprows=range(1, 124035460)
)
print(df_train)
df_train["unit_sales"] = df_train["unit_sales"].apply(np.log1p)
df_train['unit_sales'] = df_train.groupby('date')['unit_sales'].transform('mean')
df_train = df_train.drop_duplicates(subset='date')
print(df_train)
sys.exit()
# Fill gaps in dates
# Improved with the suggestion from Paulo Pinto
u_dates = df_train.date.unique()
u_stores = df_train.store_nbr.unique()
u_items = df_train.item_nbr.unique()
df_train.set_index(["date", "store_nbr", "item_nbr"], inplace=True)
df_train = df_train.reindex(
    pd.MultiIndex.from_product(
        (u_dates, u_stores, u_items),
        names=["date", "store_nbr", "item_nbr"]
    )
)

# Fill NAs
df_train.loc[:, "unit_sales"].fillna(0, inplace=True)
# Assume missing entris imply no promotion
df_train.loc[:, "onpromotion"].fillna("False", inplace=True)

# Calculate means
df_train = df_train.groupby(
    ['item_nbr', 'store_nbr', 'onpromotion']
)['unit_sales'].mean().to_frame('unit_sales')
# Inverse transform
df_train["unit_sales"] = df_train["unit_sales"].apply(np.expm1)

# Create submission
pd.read_csv(
    "../input/test.csv", usecols=[0, 2, 3, 4], dtype={'onpromotion': str}
).set_index(
    ['item_nbr', 'store_nbr', 'onpromotion']
).join(
    df_train, how='left'
).fillna(0).to_csv(
    'mean.csv.gz', float_format='%.2f', index=None, compression="gzip"
)
print(df_train.head())
# plt.plot(df_train['date'], df_train['unit_sales'])
# plt.title('Time Series - Average per Date')
# plt.xticks(rotation='vertical')
# plt.show()
# plt.style.use('seaborn-white')
# plt.figure(figsize=(13,11))
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
