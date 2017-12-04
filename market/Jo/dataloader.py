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

ept = train.loc[items[0], stores[0]][["unit_sales"]].shape[0]

print(ept)

train_split = int((ept - 1) * .8)
test_split = (ept - 1) - train_split


class SalesTrain(Dataset):
    def __len__(self):
        return int(len(items)*.5) # int(len(items) * len(stores) * .8)

    # index is store * items_count
    #        + item
    def __getitem__(self, index):
        # s = int(index // items_count)
        store = stores[0]
        # i = int(index % items_count)
        item = items[index]

        tuple_series = train.loc[item, store][["unit_sales"]]\
            .as_matrix()

        subseries = np.log1p(tuple_series[:train_split])
        target = np.log1p(tuple_series[1:train_split + 1])
        return subseries.reshape([-1]), target.reshape([-1])


class SalesTest(Dataset):
    def __len__(self):
        return int(len(items)*.5)

    def __getitem__(self, index):
        # s = int(index // items_count)
        store = stores[0]
        # i = int(index % items_count)
        item = items[index]


        tuple_series = train.loc[item, store][["unit_sales"]]\
            .as_matrix()

        subseries = np.log1p(tuple_series[train_split: -1])
        target = np.log1p(tuple_series[train_split+1:])
        return subseries.reshape([-1]), target.reshape([-1])


def train_loader():
    return torch.utils.data.DataLoader(SalesTrain(),
                                       batch_size=1, shuffle=True,
                                       num_workers=4)


def val_loader():
    return torch.utils.data.DataLoader(
        SalesTest(),
        batch_size=1, shuffle=True, num_workers=4)


# class DogsOutputDataset(Dataset):

#     def __init__(self, root_dir, transform=None):
#         self.img_path = root_dir
#         self.img_ext = ".jpg"
#         self.transform = transform

#         self.img_list = os.listdir(root_dir)

#     def __len__(self):
#         return len(self.img_list)

#     def __getitem__(self, index):
#         img = Image.open(self.img_path + self.img_list[index])
#         img = img.convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)

#         return img, self.img_list[index][:-4]


# data_transform2 = transforms.Compose([
#     transforms.Scale(CROPPED_SIZE),
#     transforms.CenterCrop(CROPPED_SIZE),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.392, 0.452, 0.476],
#                          std=[0.262, 0.257, 0.263])
# ])

# output_dataset = DogsOutputDataset(root_dir="../test/",
#                                    transform=data_transform2)

if __name__ == "__main__":
    # idx = [x[1] for x in output_dataset]
    a = SalesTrain()[0]

    print(a)
