import itertools
import pandas as pd
import numpy as np
import sys
from datetime import datetime

df_train = pd.read_csv('./prophet_predictions.csv', index_col = 0)
df_train.set_index(['date', 'yhat'])

df_train = df_train.rename(columns={'yhat': 'unit_sales'})
print(df_train.head())
print("sfsd")
df_test = pd.read_csv("../input/test.csv", usecols=[0, 1, 2, 3])

df_test.set_index(
    ['date', 'item_nbr', 'store_nbr'])
ge = pd.merge(df_test[['id', 'date', 'store_nbr', 'item_nbr']], df_train[['date', 'unit_sales']], on='date')
print(ge.head())


# .join(
#     df_train.set_index('date'), how='left', on='date'
# ).fillna(0).to_csv(
#     'what.csv.gz', float_format='%.2f', index=None, compression="gzip"
# )

header = ["id", "unit_sales"]
print(ge.head())
ge.to_csv('prophet_submission.csv', columns = header, index = False)