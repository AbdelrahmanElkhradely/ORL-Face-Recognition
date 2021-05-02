import cv2

import Load_DataSet
import SplitDataSet
import PCA
import LDA
import numpy as np
import pandas as pd
import faceVSnonfaces
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from KNN import KNNClassifier

if __name__ == '__main__':
    faceVSnonfaces.facesVSnonfaces()
    # images,labels=Load_DataSet.get_dataset()
    # test_Datamatrix, test_Labelmatrix, train_Datamatrix, train_Labelmatrix= SplitDataSet.split(images,labels)
    # test_Datamatrix = np.asarray(test_Datamatrix, dtype="int32")
    # test_Labelmatrix = np.asarray(test_Labelmatrix, dtype="int32")
    # train_Datamatrix = np.asarray(train_Datamatrix, dtype="int32")
    # train_Labelmatrix = np.asarray(train_Labelmatrix, dtype="int32")
    # # PCA.PCA(test_Datamatrix, test_Labelmatrix, train_Datamatrix, train_Labelmatrix)
    # # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    # LDA.LDA(test_Datamatrix, test_Labelmatrix, train_Datamatrix, train_Labelmatrix)
    # # KNNClassifier(train_Datamatrix.T, train_Labelmatrix, test_Datamatrix.T,test_Labelmatrix)



