import math
import numpy as np
from tqdm import tqdm

def sr_gr(img, angle):
    img_filtered = np.zeros_like(img)
    N, M = img.shape
    for i in range(0, N):
        for j in range (0, M):
            img_filtered[i,j] = img[i,j] / math.sin(angle/180 * math.pi)
    return img_filtered

def sr_gr_df(data, angles):
    return np.array([sr_gr(data[i], angles[i]) for i in range(1604)])