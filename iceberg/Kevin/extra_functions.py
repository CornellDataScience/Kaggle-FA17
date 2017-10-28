import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator


""" Generates new images by rotation, flipping, etc. """
def generateNewImages(x_train, y_train):
    dummy_dat = np.zeros((1203,75,75,1), dtype=np.float32)
    fudge_X_train = np.concatenate((x_train, dummy_dat), axis = 3)

    datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.1)

    datagen.fit(fudge_X_train)
    x_batches = fudge_X_train
    y_batches = y_train
    
    #additional_angles = np.empty([1000])
    
    batches = 0
    per_batch = 1000
    for x_batch, y_batch in datagen.flow(fudge_X_train, y_train, batch_size=per_batch, shuffle=False, seed=137):
        x_batches = np.concatenate((x_batches, x_batch), axis = 0)
        y_batches = np.concatenate((y_batches, y_batch), axis = 0)
        #additional_angles = np.append(additional_angles, x_angle_train[batches])
        #batches += 1
        break

    x_train = x_batches[:,:,:,:2]
    y_train = y_batches
    print('New features shape of training data: ')
    print(x_train.shape)
    print("New labels shape of training data: ")
    print(y_train.shape)
    return [x_train, y_train]


""" Saves the newly generated images, concatenated with the already provided images.  Essentially stores a new, 
    augmented dataset. """
def saveGeneratedImages(x_train, y_train):
    x_train_reshaped = np.reshape(x_train, (2203, 11250))
    np.savetxt("x_train_new.csv", x_train_reshaped, delimiter=",")
    np.savetxt("y_train_new.csv", y_train, delimiter=",")