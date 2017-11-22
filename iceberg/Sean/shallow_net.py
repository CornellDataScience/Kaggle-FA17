import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import random
from sklearn import preprocessing
from sklearn import model_selection
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, ZeroPadding2D, Flatten, AveragePooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.utils.np_utils import to_categorical
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df_train = pd.read_json("../input/train.json")
df_test = pd.read_json("../input/test.json")
df_train.inc_angle = df_train.inc_angle.replace('na', np.random.normal(loc=37.5, scale=3))
df_train.inc_angle = df_train.inc_angle.astype(float).fillna(np.random.normal(loc=37.5, scale=3))

# Train data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_train["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_train["band_2"]])

x_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis],
                         ((x_band1+x_band2)/2)[:, :, :, np.newaxis]], axis=-1)
x_angle_train = np.array(df_train.inc_angle)
y_train = to_categorical(df_train["is_iceberg"])
print("xtrain:", x_train.shape)

# Test data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_test["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_test["band_2"]])
x_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis],
                         ((x_band1+x_band2)/2)[:, :, :, np.newaxis]], axis=-1)
x_angle_test = np.array(df_test.inc_angle)
print("xtest:", x_test.shape)

X_train, X_valid, Y_train, Y_valid, X_train_angle, X_valid_angle = model_selection.train_test_split(x_train, y_train, x_angle_train, test_size=0.3, random_state=1)

inp_1 = Input(shape=(75, 75, 3))
inp_2 = Input(shape=[1])
weight_decay = 0.005

conv1 = Conv2D(16, (3,3), padding='same', activation='elu', kernel_initializer='glorot_normal',
              kernel_regularizer=l2(weight_decay))(BatchNormalization()(inp_1))
conv2 = Conv2D(16, (3,3), padding='same', activation='elu', kernel_initializer='glorot_normal',
              kernel_regularizer=l2(weight_decay))(BatchNormalization()(conv1))
pool1 = AveragePooling2D((2,2), strides=(2,2))(conv2)

conv3 = Conv2D(32, (3,3), padding='same', activation='elu', kernel_initializer='glorot_normal',
              kernel_regularizer=l2(weight_decay))(pool1)
conv4 = Conv2D(32, (3,3), padding='same', activation='elu', kernel_initializer='glorot_normal',
              kernel_regularizer=l2(weight_decay))(conv3)
pool2 = AveragePooling2D((2,2), strides=(2,2))(conv4)

conv5 = Conv2D(32, (3,3), padding='same', activation='elu', kernel_initializer='glorot_normal',
              kernel_regularizer=l2(weight_decay))(pool2)
conv6 = Conv2D(32, (3,3), padding='same', activation='elu', kernel_initializer='glorot_normal',
              kernel_regularizer=l2(weight_decay))(conv5)
pool3 = AveragePooling2D((2,2), strides=(2,2))(conv6)

conv7 = Conv2D(32, (3,3), padding='same', activation='elu', kernel_initializer='glorot_normal',
              kernel_regularizer=l2(weight_decay))(pool3)
conv8 = Conv2D(32, (3,3), padding='same', activation='elu', kernel_initializer='glorot_normal',
              kernel_regularizer=l2(weight_decay))(conv7)
pool4 = AveragePooling2D((2,2), strides=(2,2))(conv8)

flat = Flatten()(pool4)
print(inp_2)
comb_feats = (Concatenate()([flat, BatchNormalization()(inp_2)]))

dense1 = Dense(64, activation='relu', kernel_initializer='glorot_normal',
              kernel_regularizer=l2(weight_decay))(comb_feats)
drop1 = Dropout(0.8)(dense1)
dense2 = Dense(64, activation='relu', kernel_initializer='glorot_normal',
              kernel_regularizer=l2(weight_decay))(drop1)
drop2 = Dropout(0.8)(dense2)
out = Dense(2, activation='softmax')(drop2)

model = Model(inputs=[inp_1,inp_2], outputs=out)
optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
model.summary()

model.fit([X_train, X_train_angle], Y_train, batch_size=64, epochs=70, validation_data=([X_valid, X_valid_angle], Y_valid), verbose=1)

preds = model.predict([x_test, x_angle_test], verbose=1)

submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': preds[:,1]})
submission.head(200)

submission.to_csv("./submission.csv", index=False)
