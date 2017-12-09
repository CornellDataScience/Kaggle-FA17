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

#Kevin's CNN architecture
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

#Fill in the angular na's with a value most likely between (29.1, 46) degrees
print("Filling in 0's")
train_angle_2 = np.copy(train_df.inc_angle)
for i in range(0,train_angle_2.size):
    if str(train_angle_2[i]) == 'na':
        train_angle_2[i] = np.random.normal(37.5, 3.75, 1)[0]
train_df.inc_angle = train_angle_2
x_angle_train = np.array(train_df.inc_angle)
x_angle_test = np.array(test_df.inc_angle)   
y_train = to_categorical(train_df["is_iceberg"])

x_train, x_val, x_angle_train, x_angle_val, y_train, y_val = train_test_split(train_images, x_angle_train, y_train, train_size=0.7)

print('Train', x_train.shape, y_train.shape)
print('Validation', x_val.shape, y_val.shape) 

#0.006 is the best
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

#Training
print("Training")
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)
model.fit([x_train, x_angle_train], y_train, batch_size = 64, validation_data = ([x_val, x_angle_val], y_val), 
          epochs = 35, shuffle = True, callbacks=[early_stopping])

#Predicting
print("predicting")
test_predictions = model.predict([test_images, x_angle_test])

pred_df = test_df[['id']].copy()
pred_df['is_iceberg'] = test_predictions[:,1]
print("creating csv")
pred_df.to_csv('predictions_9.csv', index = False)