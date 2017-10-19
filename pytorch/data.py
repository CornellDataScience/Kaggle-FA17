from torch.utils.data import Dataset, DataLoader
import pandas as pd
from skimage import io, transform
import os
import torch
from torchvision import transforms, utils
import numpy as np
import torchvision.datasets as datasets
from PIL import Image
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

CROPPED_SIZE = 320 #224


class DogsDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        tmp_df = pd.read_csv(csv_file)
        assert tmp_df['id'].apply(lambda x: os.path.isfile(root_dir + x + ".jpg")).all(), \
            "Some images referenced in the CSV file were not found"

        self.img_path = root_dir
        self.img_ext = ".jpg"
        self.transform = transform

        self.X_train = tmp_df['id'][:9000]
        self.y_train = le.fit_transform(
            tmp_df['breed'])[:9000]

    def __len__(self):
        return len(self.X_train.index)

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = self.y_train[index]
        return img, label


class DogsDatasetTest(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        tmp_df = pd.read_csv(csv_file)
        assert tmp_df['id'].apply(lambda x: os.path.isfile(root_dir + x + ".jpg")).all(), \
            "Some images referenced in the CSV file were not found"

        self.img_path = root_dir
        self.img_ext = ".jpg"
        self.transform = transform

        self.X_train = tmp_df['id'][9000:].reset_index(drop=True)
        self.y_train = le.fit_transform(
            tmp_df['breed'])[9000:]

    def __len__(self):
        return len(self.X_train.index)

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.y_train[index]
        return img, label


def dogsDataset(size=CROPPED_SIZE):
    transform = transforms.Compose([
        transforms.RandomSizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.392, 0.452, 0.476],
                             std=[0.262, 0.257, 0.263])
    ])
    return DogsDataset(csv_file="../labels.csv", root_dir="../train/",
                       transform=transform)


def dogsDatasetTest(size=CROPPED_SIZE):
    transform = transforms.Compose([
        transforms.Scale(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.392, 0.452, 0.476],
                             std=[0.262, 0.257, 0.263])
    ])
    return DogsDatasetTest(csv_file="../labels.csv", root_dir="../train/",
                           transform=transform)


def train_loader(size=CROPPED_SIZE):
    return torch.utils.data.DataLoader(dogsDataset(size),
                                       batch_size=64, shuffle=True,
                                       num_workers=4)


def val_loader(size=CROPPED_SIZE):
    return torch.utils.data.DataLoader(
        dogsDatasetTest(size),
        batch_size=64, shuffle=True, num_workers=4)


class DogsOutputDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.img_path = root_dir
        self.img_ext = ".jpg"
        self.transform = transform

        self.img_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.img_list[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, self.img_list[index][:-4]


data_transform2 = transforms.Compose([
    transforms.Scale(CROPPED_SIZE),
    transforms.CenterCrop(CROPPED_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.392, 0.452, 0.476],
                         std=[0.262, 0.257, 0.263])
])

output_dataset = DogsOutputDataset(root_dir="../test/",
                                   transform=data_transform2)

if __name__ == "__main__":
    # idx = [x[1] for x in output_dataset]
    a = dogsDataset()[0]

    print(a)

    print(le.transform(["affenpinscher", "afghan_hound"]))
