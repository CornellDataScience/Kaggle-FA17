from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torch
from torchvision import transforms, utils
import numpy as np
import torchvision.datasets as datasets
from sklearn.preprocessing import LabelEncoder


train = pd.read_pickle("lstm_train_pickle.pkl")

train.set_index(["item_nbr", "store_nbr"], inplace=True)
train = train.sort_index()

items = train.index.levels[0]
stores = train.index.levels[1]

items_count = len(items)
stores_count = len(stores)

class SalesAll(Dataset):
    def __len__(self):
        return int(len(items) * len(stores))

    # index is store * items_count
    #        + item
    def __getitem__(self, index):
        s = int(index // items_count)
        store = stores[s]
        i = int(index % items_count)
        item = items[i]

        tuple_series = train.loc[item, store][["unit_sales"]]\
            .as_matrix()

        subseries = np.log1p(tuple_series)
        return store, item, subseries.reshape([-1])

def all_loader():
    return torch.utils.data.DataLoader(
        SalesAll(),
        batch_size=1, shuffle=True, num_workers=4)
