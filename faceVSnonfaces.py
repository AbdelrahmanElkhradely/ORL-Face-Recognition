import cv2

import Load_DataSet
import SplitDataSet
import PCA
import LDAfornonfaces
import numpy as np
import pandas as pd
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from KNN import KNNClassifier

def facesVSnonfaces():
    BASE = './facesvsnonfaces/'
    filepaths = []
    for s_i in np.os.listdir(BASE):
        if s_i != 'README':
            for filename in np.os.listdir(BASE + s_i):
                filepaths.append(BASE + s_i + '/' + filename)
    df = pd.DataFrame({'filepaths': filepaths})
    images = []
    labels = []
    for filepath in df['filepaths']:
        images.append(cv2.imread(filepath, 0).flatten())
        x = filepath.split("/")
        x = x[2]
        x = x.replace('s', '')
        labels.append(x)
    test_Datamatrix = []
    test_Labelmatrix = []
    train_Datamatrix = []
    train_Labelmatrix = []
    from sklearn.model_selection import train_test_split
    train_Datamatrix, test_Datamatrix, train_Labelmatrix, test_Labelmatrix = train_test_split(
        images, labels, test_size=0.5, random_state=42)
    test_Datamatrix = np.asarray(test_Datamatrix, dtype="int32")
    test_Labelmatrix = np.asarray(test_Labelmatrix, dtype="int32")
    train_Datamatrix = np.asarray(train_Datamatrix, dtype="int32")
    train_Labelmatrix = np.asarray(train_Labelmatrix, dtype="int32")
    LDAfornonfaces.LDAfornonfaces(test_Datamatrix, test_Labelmatrix, train_Datamatrix, train_Labelmatrix)
