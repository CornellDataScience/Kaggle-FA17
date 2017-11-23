# The first attempt of the VGG-16 like neural network
# Got log loss 0.26
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from sklearn import preprocessing
from sklearn import model_selection

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# simply filling missing incidence angle with 0
print("data preprocessing...")
df_train = pd.read_json("../input/train.json")
df_test = pd.read_json("../input/test.json")
df_train.inc_angle = df_train.inc_angle.replace('na', 0)
df_train.inc_angle = df_train.inc_angle.astype(float).fillna(0.0)

print("setting up train and test...")
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_train["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_train["band_2"]])
x_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
y_train = np.array(df_train["is_iceberg"])
print("xtrain:", x_train.shape)
# Test data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_test["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df_test["band_2"]])
x_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
print("xtest:", x_test.shape)
X_train, X_valid, Y_train, Y_valid = model_selection.train_test_split(x_train, y_train, test_size=0.3, random_state=1)

# Build a VGG-16 like model with fewer layers
print("building neural network architecture...")
inp = Input(shape=(75, 75, 2))

conv1 = Conv2D(32, (3,3), padding='same', activation='relu', kernel_initializer='glorot_normal')(inp)
conv2 = Conv2D(32, (3,3), padding='same', activation='relu', kernel_initializer='glorot_normal')(conv1)
pool1 = MaxPooling2D((2,2), strides=(2,2))(conv2)

conv3 = Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='glorot_normal')(pool1)
conv4 = Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='glorot_normal')(conv3)
pool2 = MaxPooling2D((2,2), strides=(2,2))(conv4)

conv5 = Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer='glorot_normal')(pool2)
conv6 = Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer='glorot_normal')(conv5)
conv7 = Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer='glorot_normal')(conv6)
pool3 = MaxPooling2D((2,2), strides=(2,2))(conv7)

flat = Flatten()(pool3)

dense1 = Dense(4096, activation='relu', kernel_initializer='glorot_normal')(flat)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(4096, activation='relu', kernel_initializer='glorot_normal')(drop1)
drop2 = Dropout(0.5)(dense2)
out = Dense(1, activation='sigmoid')(drop2)

print("setting up the model...")
model = Model(inputs=inp, outputs=out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
model.summary()

print("training...")
model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_valid, Y_valid), verbose=1)

print("predicting...")
preds = model.predict(x_test, verbose=1)

print("writing to csv...")
submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': preds.reshape((preds.shape[0]))})
submission.head(100)
submission.to_csv("./submission.csv", index=False)
