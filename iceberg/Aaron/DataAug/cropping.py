from random import *
import numpy as np
import pandas as pd
from tqdm import tqdm

def crop(img):
    x = randint(0,15)
    y = randint(0,15)
    newimg = np.empty([60, 60])
    newimg = img[x:x+60, y: y+60]
    return newimg

def croptest(img):
    newimg = np.empty([60, 60])
    newimg = img[7:67, 7:67]
    return newimg

def crop_aug(images, angles, values, number):
    newimages = []
    newangles = []
    newvalues = []
    for i in tqdm(range(np.size(images,0))):
        for j in range(number):
            newimages.append(crop(images[i]))
            newangles.append(angles[i])
            newvalues.append(values[i])
    return np.array(newimages), np.array(newangles), np.array(newvalues)

def crop_test(test):
    newtest = []
    for i in tqdm(range(np.size(test,0))):
        newtest.append(crop(test[i]))
    return np.array(newtest)

def test_condense(test, number):
    testresults = []
    for i in tqdm((range(np.size(test, 0)))/number):
        sum = 0;
        for j in range(number):
            sum += test[i*number + j]
        testresults.append(sum/number)
    return np.aray(testresults)