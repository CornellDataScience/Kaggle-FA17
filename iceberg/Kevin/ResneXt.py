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
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Dense, Flatten, Activation, LeakyReLU, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.layers import average, Input, Concatenate, Lambda
from extra_functions import *
from keras import layers
from keras.optimizers import Adam
from keras.constraints import max_norm


cardinality = 32

def load_and_format(in_path):
    out_df = pd.read_json(in_path)
    out_images = out_df.apply(lambda c_row: [np.stack([c_row['band_1'],c_row['band_2']], -1).reshape((75,75,2))],1)
    out_images = np.stack(out_images).squeeze()
    return out_df, out_images


def add_common_layers(cnn):
    #cnn = BatchNormalization(momentum=0.99)(cnn)
    cnn = Activation('relu')(cnn)
    return cnn


def grouped_convolution(cnn, nb_channels, _strides):
    # when `cardinality` == 1 this is just a standard convolution
    if cardinality == 1:
        return Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(cnn)
        
    assert not nb_channels % cardinality
    _d = nb_channels // cardinality

    # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
    # and convolutions are separately performed within each group
    groups = []
    for j in range(cardinality):
        group = Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(cnn)
        groups.append(Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
            
    # the grouped convolutional layer concatenates them as the outputs of the layer
    cnn = layers.concatenate(groups)

    return cnn


def residual_block(cnn, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = cnn

        # we modify the residual building block as a bottleneck design to make the network more economical
        cnn = Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(cnn)
        cnn = add_common_layers(cnn)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        cnn = grouped_convolution(cnn, nb_channels_in, _strides=_strides)
        cnn = add_common_layers(cnn)

        cnn = Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(cnn)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        #cnn = BatchNormalization(momentum=0.99)(cnn)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            #shortcut = BatchNormalization(momentum=0.99)(shortcut)

        cnn = layers.add([shortcut, cnn])
        cnn = Activation('relu')(cnn)

        return cnn


dir_path = path.abspath(path.join('__file__',"../.."))
train_path = dir_path + "/train.json"
test_path = dir_path + "/test.json"

train_df, train_images = load_and_format(train_path)
test_df, test_images = load_and_format(test_path)
train_df.inc_angle = train_df.inc_angle.replace('na', 0)
train_df.inc_angle = train_df.inc_angle.astype(float).fillna(0.0)
x_angle_train = np.array(train_df.inc_angle)
x_angle_test = np.array(test_df.inc_angle)   
y_train = to_categorical(train_df["is_iceberg"])

x_train, x_val, x_angle_train, x_angle_val, y_train, y_val = train_test_split(train_images, x_angle_train, y_train, train_size=0.7)

print('Train', x_train.shape, y_train.shape)
print('Validation', x_val.shape, y_val.shape) 

#0.006
weight_decay = 0.006

image_input = Input(shape=(75, 75, 2), name="image")
angle_input = Input(shape=[1], name='angle')

cnn = BatchNormalization(momentum=0.99)(image_input)

cnn = Conv2D(32, kernel_size=(2,2), padding = 'same')(cnn)
cnn = add_common_layers(cnn)
cnn = AveragePooling2D((2, 2))(cnn)

cnn = residual_block(cnn, 32, 32)
cnn = residual_block(cnn, 32, 32)
cnn = residual_block(cnn, 32, 32)

cnn = AveragePooling2D((2, 2))(cnn)

cnn = residual_block(cnn, 32, 32)
cnn = residual_block(cnn, 32, 32)
cnn = residual_block(cnn, 32, 32)

cnn = AveragePooling2D((2, 2))(cnn)

cnn = residual_block(cnn, 32, 32)
cnn = residual_block(cnn, 32, 32)
cnn = residual_block(cnn, 32, 32)

cnn = AveragePooling2D((2, 2))(cnn)

cnn = residual_block(cnn, 32, 32)
cnn = residual_block(cnn, 32, 32)
cnn = residual_block(cnn, 32, 32)

cnn = AveragePooling2D((2, 2))(cnn)
cnn = Dropout(0.2)(cnn)

cnn = Flatten()(cnn)
cnn = Concatenate()([cnn, BatchNormalization()(angle_input)])

#kernel_regularizer=l2(weight_decay)

cnn = Dense(100, activation='relu')(cnn)
cnn = Dropout(0.2)(cnn)

cnn = Dense(50, activation='relu')(cnn)
cnn = Dropout(0.2)(cnn)

output = Dense(2, activation='softmax')(cnn)


optimizer = Adam(lr=0.001)
model = Model(inputs=[image_input, angle_input], outputs=output)
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 8)

model.fit([x_train, x_angle_train], y_train, batch_size = 64, validation_data = ([x_val, x_angle_val], y_val), 
          epochs = 40, shuffle = True, callbacks=[early_stopping])

print("predicting")
test_predictions = model.predict([test_images, x_angle_test])


pred_df = test_df[['id']].copy()
pred_df['is_iceberg'] = test_predictions[:,1]
print("creating csv")
pred_df.to_csv('predictions.csv', index = False)