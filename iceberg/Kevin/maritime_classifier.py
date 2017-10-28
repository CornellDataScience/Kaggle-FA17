import math
import numpy as np
import pandas as pd
from skimage.util.montage import montage2d
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Dense, Flatten, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from extra_functions import *

def load_and_format(in_path):
    out_df = pd.read_json(in_path)
    out_images = out_df.apply(lambda c_row: [np.stack([c_row['band_1'],c_row['band_2']], -1).reshape((75,75,2))],1)
    out_images = np.stack(out_images).squeeze()
    return out_df, out_images

train_df, train_images = load_and_format('/Users/kevinluo/Desktop/Iceberg_Data/train.json')
test_df, test_images = load_and_format('/Users/kevinluo/Desktop/Iceberg_Data/test.json')

print("reading data")
x_train = pd.read_csv('x_train.csv', header=None)
x_train = x_train.values
y_train = pd.read_csv('y_train.csv', header=None)
y_train = y_train.values
x_val = pd.read_csv('x_val.csv', header=None)
x_val = x_val.values
y_val = pd.read_csv('y_val.csv', header=None)
y_val = y_val.values

print("reshaping data")
x_train = np.reshape(x_train, (3366, 75, 75, 2))
x_val = np.reshape(x_val, (482, 75, 75, 2))

print('Train', x_train.shape, y_train.shape)
print('Validation', x_val.shape, y_val.shape)

cnn = Sequential()

#Preprocess Data with Mean Normalization
cnn.add(BatchNormalization(input_shape=(75, 75, 2)))

#Add First Convolutional Layer
print("Adding First Convolutional Layer")
cnn.add(ZeroPadding2D((1,1)))
cnn.add(Conv2D(64, kernel_size=(3,3), input_shape=(75, 75, 2), activation='relu', kernel_initializer='TruncatedNormal'))
cnn.add(ZeroPadding2D((1,1)))
cnn.add(Conv2D(64, kernel_size=(3,3), input_shape=(75, 75, 2), activation='relu', kernel_initializer='TruncatedNormal'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

#Add Second Convolutional Layer
print("Adding Second Convolutional Layer")
cnn.add(ZeroPadding2D((1,1)))
cnn.add(Conv2D(128, kernel_size=(3,3), activation='relu', kernel_initializer='TruncatedNormal'))
cnn.add(ZeroPadding2D((1,1)))
cnn.add(Conv2D(128, kernel_size=(3,3), activation='relu', kernel_initializer='TruncatedNormal'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

#Add Third Convolutional Layer
print("Adding Third Convolutional Layer")
cnn.add(ZeroPadding2D((1,1)))
cnn.add(Conv2D(256, kernel_size=(3,3), activation='relu', kernel_initializer='TruncatedNormal'))
cnn.add(ZeroPadding2D((1,1)))
cnn.add(Conv2D(256, kernel_size=(3,3), activation='relu', kernel_initializer='TruncatedNormal'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

#Add Fourth Convolutional Layer
print("Adding Fourth Convolutional Layer")
cnn.add(ZeroPadding2D((1,1)))
cnn.add(Conv2D(512, kernel_size=(3,3), activation='relu', kernel_initializer='TruncatedNormal'))
cnn.add(ZeroPadding2D((1,1)))
cnn.add(Conv2D(512, kernel_size=(3,3), activation='relu', kernel_initializer='TruncatedNormal'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

#Add Fifth Convolutional Layer
#print("Adding Fifth Convolutional Layer")
#cnn.add(Conv2D(120, kernel_size=(3,3), activation='relu', kernel_initializer='TruncatedNormal'))
#cnn.add(MaxPooling2D(pool_size=(2,2)))

#Add Fully-Connected Layer
print("Adding Fully-Connected Layer")
#Flatten so that the data can pass through a FC Layer
cnn.add(Flatten())
cnn.add(Dense(100, activation='relu', kernel_initializer='TruncatedNormal'))
#cnn.add(Dropout(0.3))

#Add Output Layer
print("Adding Output Layer")
cnn.add(Dense(2, activation='softmax', kernel_initializer='TruncatedNormal'))

#Define loss function and optimizer
cnn.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

cnn.summary()
print("Training")
cnn.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 15, shuffle = True)



"""print("predicting")
test_predictions = cnn.predict(test_images)

pred_df = test_df[['id']].copy()
pred_df['is_iceberg'] = test_predictions[:,1]
print("creating csv")
pred_df.to_csv('predictions.csv', index = False) """