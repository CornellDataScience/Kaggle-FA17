import pandas as pd
import numpy as np

dtypes = {'id': 'int64', 'item_nbr': 'int32', 'store_nbr': 'int8'}

df_train = pd.read_csv("../csv/train.csv",
                       usecols=[1, 2, 3, 4],
                       dtype=dtypes,
                       parse_dates=['date'],
                       skiprows=range(1, 101688779))

df_train.loc[(df_train.unit_sales < 0), 'unit_sales'] = 0
df_train['unit_sales'] =  df_train['unit_sales'].apply(np.log1p)

u_dates = df_train.date.unique()
u_stores = df_train.store_nbr.unique()
u_items = df_train.item_nbr.unique()
df_train.set_index(['date', 'store_nbr', 'item_nbr'], inplace=True)
df_train = df_train.reindex(
    pd.MultiIndex.from_product(
        (u_dates, u_stores, u_items),
        names=['date','store_nbr','item_nbr']
    )
)

del u_dates, u_stores, u_items

df_train.loc[:, 'unit_sales'].fillna(0, inplace=True)
df_train.reset_index(inplace=True)
lastdate = df_train.iloc[df_train.shape[0]-1].date

test = pd.read_csv('../input/test.csv', dtype=dtypes, parse_dates=['date'])

ma_dw = train[['item_nbr','store_nbr','dow','unit_sales']].groupby(['item_nbr','store_nbr','dow'])['unit_sales'].mean().to_frame('madw')
ma_dw.reset_index(inplace=True)
ma_wk = ma_dw[['item_nbr','store_nbr','madw']].groupby(['store_nbr', 'item_nbr'])['madw'].mean().to_frame('mawk')
ma_wk.reset_index(inplace=True)