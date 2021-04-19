import Load_DataSet
import SplitDataSet
import PCA
import numpy as np

if __name__ == '__main__':
    images,labels=Load_DataSet.get_dataset()
    test_Datamatrix, test_Labelmatrix, train_Datamatrix, train_Labelmatrix= SplitDataSet.split(images,labels)
    test_Datamatrix = np.asarray(test_Datamatrix, dtype="int32")
    test_Labelmatrix = np.asarray(test_Labelmatrix, dtype="int32")
    train_Datamatrix = np.asarray(train_Datamatrix, dtype="int32")
    train_Labelmatrix = np.asarray(train_Labelmatrix, dtype="int32")
    PCA.PCA(test_Datamatrix, test_Labelmatrix, train_Datamatrix, train_Labelmatrix)



