from multiprocessing.dummy import Pool as ThreadPool

import pandas as pd
train = pd.read_pickle("lstm_train_pickle.pkl")
train = train[train["date"] < "2017-01-01"]

import numpy as np
train2 = train.set_index(["item_nbr", "store_nbr"], drop=True)

items = train2.index.levels[0]
stores = train2.index.levels[1]

def get_series(item):
    print(item)
    return train2.loc[item, 44][["unit_sales"]].as_matrix().reshape([-1])

pool = ThreadPool(4) 
results = pool.map(get_series, items)

import pickle
f = open('test.pickle', 'wb')
pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
f.close()