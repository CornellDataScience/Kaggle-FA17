import itertools

import pandas as pd
import numpy as np

from datetime import datetime

df_train = pd.read_csv(
    './input/train.csv', usecols=[1, 2, 3, 4],
    converters={'unit_sales': lambda u: float(u) if float(u) > 0 else 0},
    skiprows=range(1, 124035460)
)

#weekday_ratio = [0.967, 0.895, 0.9336, 0.8471, 0.9262, 1.1697, 1.2572]
weekday_ratio = [0.967, 0.925, 0.9336, 0.8471, 0.9462, 1.097, 1.1572]
def weekday_unit_sale_adjustment(row):
    weekday = datetime.strptime(row['date'], "%Y-%m-%d").weekday()
    new_unit_price = row['unit_sales'] * weekday_ratio[weekday]
    return new_unit_price

# Adjust all sales according to their weekday
df_train["unit_sales"] = df_train.apply(lambda row: weekday_unit_sale_adjustment(row), axis=1)

# log transform
df_train["unit_sales"] = df_train["unit_sales"].apply(np.log1p)
print("After log transform")
print(df_train.head())


u_dates = df_train.date.unique()
u_stores = df_train.store_nbr.unique()
u_items = df_train.item_nbr.unique()
df_train.set_index(["date", "store_nbr", "item_nbr"], inplace=True)
print("set index")
print(df_train.head())
df_train = df_train.reindex(
    pd.MultiIndex.from_product(
        (u_dates, u_stores, u_items),
        names=["date","store_nbr", "item_nbr"]
    )
)
print("reindex")
print(df_train.head())
# Fill NAs
df_train.loc[:, "unit_sales"].fillna(0, inplace=True)
# Assume missing entries imply no promotion
# df_train.loc[:, "onpromotion"].fillna("False", inplace=True)

# Calculate means
df_train = df_train.groupby(
    ['item_nbr', 'store_nbr']
)['unit_sales'].mean().to_frame('unit_sales')
# Inverse transform
df_train["unit_sales"] = df_train["unit_sales"].apply(np.expm1)
print(df_train.head())
# Create submission
pd.read_csv(
    "./input/test.csv", usecols=[0, 2, 3]
).set_index(
    ['item_nbr', 'store_nbr']
).join(
    df_train, how='left'
).fillna(0).to_csv(
    'normal_mean2.csv.gz', float_format='%.2f', index=None, compression="gzip"
)
