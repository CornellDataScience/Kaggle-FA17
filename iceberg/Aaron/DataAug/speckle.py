from random import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import expon
import sys
sys.path.insert(0, '../Filters')
import median

def speckle_noise(img):
    row,col = img.shape
    r = expon.rvs(size= (row*col))
    r = r.reshape(row,col)        
    noisy = img + np.log(r)
    return noisy

#Precondition: images is of shape [:,:,:,2]
def speckle_aug(images, angles, values, number):
    newimages1 = []
    newimages2 = []
    newangles = []
    newvalues = []
    x1 = median.median_df(images[:,:,:,0])
    x2 = median.median_df(images[:,:,:,1])
    for i in tqdm(range(np.size(images,0))):
        for j in range(number):
            newimages1.append(speckle_noise(x1[i]))
            newimages2.append(speckle_noise(x2[i]))
            newangles.append(angles[i])
            newvalues.append(values[i])
    newimages1 = np.array(newimages1)
    newimages2 = np.array(newimages2)
    return np.concatenate((newimages1[:,:,:,np.newaxis], newimages2[:,:,:,np.newaxis]), axis = 3), np.array(newangles), np.array(newvalues)