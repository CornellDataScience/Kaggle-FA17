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


class DogsDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        tmp_df = pd.read_csv(csv_file)
        assert tmp_df['id'].apply(lambda x: os.path.isfile(root_dir + x + ".jpg")).all(), \
            "Some images referenced in the CSV file were not found"

        self.img_path = root_dir
        self.img_ext = ".jpg"
        self.transform = transform

        self.X_train = tmp_df['id']
        self.y_train = le.fit_transform(
            tmp_df['breed'])

    def __len__(self):
        return len(self.X_train.index)

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = self.y_train[index]
        return img, label

data_transform = transforms.Compose([
    transforms.Scale(68),
    transforms.RandomSizedCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.392, 0.452, 0.476],
                         std=[0.262, 0.257, 0.263])
])

transform_test = transforms.Compose([
    transforms.Scale(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.392, 0.452, 0.476],
                         std=[0.262, 0.257, 0.263])
])

dogsDataset = DogsDataset(csv_file="labels.csv", root_dir="train/",
                          transform=data_transform)

train_loader = torch.utils.data.DataLoader(dogsDataset,
                                             batch_size=64, shuffle=True,
                                             num_workers=4, pin_memory = True)

val_loader = torch.utils.data.DataLoader(
    dogsDataset,
    batch_size=64, shuffle=True, num_workers=4, pin_memory = True)


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
    transforms.Scale(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.392, 0.452, 0.476],
                         std=[0.262, 0.257, 0.263])
])

output_dataset = DogsOutputDataset(root_dir="test/",
                          transform=data_transform2)

if __name__ == "__main__":
    idx = [x[1] for x in output_dataset]

    print(idx)
