import numpy as np
import pandas as pd

import cv2
def get_dataset():
    BASE = './archive/'
    filepaths = []
    for s_i in np.os.listdir(BASE):
        if s_i != 'README':
            for filename in np.os.listdir(BASE + s_i):
                filepaths.append(BASE + s_i + '/' + filename)
    df = pd.DataFrame({'filepaths':filepaths})
    images = []
    labels=[]
    for filepath in df['filepaths']:
        images.append(cv2.imread(filepath, 0).flatten())
        x = filepath.split("/")
        x=x[2]
        x=x.replace('s', '')
        labels.append(x)
    return images,labels
