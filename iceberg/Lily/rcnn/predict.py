import os
import numpy as np # linear algebra
import pandas as pd # data processing
import datetime as dt

import preprocessing
import utils
import rcnn

import keras.layers
import keras.optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

# load data
train, train_bands = utils.read_jason(file='train.json', loc='../')
test, test_bands = utils.read_jason(file='test.json', loc='../')

generator = preprocessing.ObjectDetectionGenerator()

classes = {
    "rbc": 1,
    "not": 2
}

generator = generator.flow(train, classes)

# create an RCNN instance
image = keras.layers.input((75, 75, 2))
model = rcnn(image, classes=len(classes) + 1)

# define the optimizer and compile
optimizer = keras.optimizers.Adam(0.001)
model.compile(optimizer)

# call backs
earlystop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, min_delta=1e-4, mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=40, verbose=1, epsilon=1e-4, mode='min')
model_chk = ModelCheckpoint(monitor='val_loss', filepath=weights_file, save_best_only=True, save_weights_only=True, mode='min')
callbacks = [earlystop, reduce_lr_loss, model_chk, TensorBoard(log_dir='../logs')]

# train the model
model.fit_generator(generator, 256, epochs=32, callbacks=callbacks)

# make prediction
x = generator.next()[0]
pred = model.predict(x)
pred = np.squeeze(pred, axis=-1)
            
subm = pd.DataFrame({'id': ids, target: pred})
subm.to_csv('rcnn_sub.csv', index=False, float_format='%.6f')
