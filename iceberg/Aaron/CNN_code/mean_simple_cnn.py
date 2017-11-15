#Mean is slightly better ~91% w/o gaussian angles
#Mean is slightly worse ~87% w/ gaussian angles
import math
import os.path as path
import numpy as np
import pandas as pd
from skimage.util.montage import montage2d
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Dense, Flatten, ZeroPadding2D, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.layers import average, Input, Concatenate

#Import mean filter
import sys
sys.path.insert(0, '../Filters')
sys.path.insert(0, '../../Kevin')

from extra_functions import *
import mean

#Import date time
import datetime

def load_and_format(in_path):
    out_df = pd.read_json(in_path)
    out_images = out_df.apply(lambda c_row: [np.stack([c_row['band_1'],c_row['band_2']], -1).reshape((75,75,2))],1)
    out_images = np.stack(out_images).squeeze()
    return out_df, out_images

dir_path = path.abspath(path.join('__file__',"../.."))
train_path = "../../train.json"
test_path =  "../../test.json"

train_df, train_images = load_and_format(train_path)
test_df, test_images = load_and_format(test_path)
#train_df.inc_angle = train_df.inc_angle.replace('na', 0)
#train_df.inc_angle = train_df.inc_angle.astype(float).fillna(0.0)
train_df.inc_angle = train_df.inc_angle.replace('na', np.random.normal(37.5, 7.5, 1)[0])
train_df.inc_angle = train_df.inc_angle.astype(float).fillna(np.random.normal(37.5, 7.5, 1)[0])
x_angle_train = np.array(train_df.inc_angle)
x_angle_test = np.array(test_df.inc_angle)   
y_train = to_categorical(train_df["is_iceberg"])

#Mean Filter on Train Images
train_images[:, :, :, 0] = mean.mean_filter_df(train_images[:, :, :, 0])
train_images[:, :, :, 1] = mean.mean_filter_df(train_images[:, :, :, 1])

#Mean on Test Images
test_images[:, :, :, 0] = mean.mean_filter_df(test_images[:, :, :, 0])
test_images[:, :, :, 1] = mean.mean_filter_df(test_images[:, :, :, 1])

x_train, x_val, x_angle_train, x_angle_val, y_train, y_val = train_test_split(train_images, x_angle_train, y_train, train_size=0.7)

print('Train', x_train.shape, y_train.shape)
print('Validation', x_val.shape, y_val.shape) 

#0.006 ~ 0.20 LB, 0.9108
#0.008 0.9 ish
#0.004 0.9 but slightly better
#0.005 also bad
weight_decay = 0.006

image_input = Input(shape=(75, 75, 2), name="image")
angle_input = Input(shape=[1], name='angle')

cnn = BatchNormalization(momentum=0.99)(image_input)

cnn = Conv2D(32, kernel_size=(2,2), padding = 'same', kernel_regularizer=l2(weight_decay))(cnn)
cnn = Activation('relu')(cnn)
cnn = MaxPooling2D(pool_size=(3,3))(cnn)

cnn = Conv2D(64, kernel_size=(3,3), padding = 'same', kernel_regularizer=l2(weight_decay))(cnn)
cnn = Activation('relu')(cnn)
cnn = MaxPooling2D(pool_size=(2,2))(cnn)

cnn = Conv2D(64, kernel_size=(3,3), padding = 'same', kernel_regularizer=l2(weight_decay))(cnn)
cnn = Activation('relu')(cnn)
cnn = MaxPooling2D(pool_size=(2,2))(cnn)

cnn = Conv2D(64, kernel_size=(3,3), padding = 'same', kernel_regularizer=l2(weight_decay))(cnn)
cnn = Activation('relu')(cnn)
cnn = MaxPooling2D(pool_size=(2,2))(cnn)

cnn = Flatten()(cnn)
cnn = Concatenate()([cnn, BatchNormalization()(angle_input)])

cnn = Dense(100, activation='relu', kernel_regularizer=l2(weight_decay))(cnn)

output = Dense(2, activation='softmax')(cnn)


model = Model(inputs=[image_input, angle_input], outputs=output)
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
print("Training")
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)
model.fit([x_train, x_angle_train], y_train, batch_size = 64, validation_data = ([x_val, x_angle_val], y_val), 
          epochs = 35, shuffle = True, callbacks=[early_stopping])


print("predicting")
test_predictions = model.predict([test_images, x_angle_test])

pred_df = test_df[['id']].copy()
pred_df['is_iceberg'] = test_predictions[:,1]
print("creating csv")
filename = "Prediction_" + str(datetime.datetime.now()) + ".csv"
print(filename)
pred_df.to_csv(filename, index = False)