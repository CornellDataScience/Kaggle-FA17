from random import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import expon
import sys
sys.path.insert(0, '../Filters')
import median

def speckle_noise(img):
    r = np.full(img.shape, expon.rvs(size = 1)[0])
    return img*r

def speckle_aug(images, angles, values, number):
    newimages = []
    newangles = []
    newvalues = []
    for i in tqdm(range(np.size(images,0))):
        for j in range(number):
            newimages.append(speckle_noise(median.med_mult(images[i])))
            newangles.append(angles[i])
            newvalues.append(values[i])
    return np.array(newimages), np.array(newangles), np.array(newvalues)