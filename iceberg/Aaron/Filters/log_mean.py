import numpy as np
import math
from tqdm import tqdm

def log_filter(img):
    N, M = img.shape
    img_filtered = np.zeros_like(img)
    for i in range(0,N):
        for j in range(0,M):
            value = 0
            if(img[i,j] < 0):
                value = img[i,j]
            if(img[i,j] >= 0):
                value = math.log(1 + img[i,j])
            img_filtered[i,j] = value
            
    return img_filtered

def log_df(data):
    return np.array([log_filter(band) for band in tqdm(data)])