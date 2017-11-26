from keras.models import load_model
import numpy as np
import pandas as pd
import os.path as path


""" Reads in data. """
def load_and_format(in_path):
    out_df = pd.read_json(in_path)
    out_images = out_df.apply(lambda c_row: [np.stack([c_row['band_1'],c_row['band_2']], -1).reshape((75,75,2))],1)
    out_images = np.stack(out_images).squeeze()
    return out_df, out_images


#obtain locations of the data
dir_path = path.abspath(path.join('__file__',"../.."))
test_path = dir_path + "/test.json"
test_df, test_images = load_and_format(test_path)
x_angle_test = np.array(test_df.inc_angle)   

#load in saved model
print('loading saved model')
model = load_model('models/resnet_batchNorm_models/newmod9_0.1630.hdf5')

print("predicting")
test_predictions = model.predict([test_images, x_angle_test])

pred_df = test_df[['id']].copy()
pred_df['is_iceberg'] = test_predictions[:,1]
print("creating csv")
pred_df.to_csv('new_predictions_9.csv', index = False)