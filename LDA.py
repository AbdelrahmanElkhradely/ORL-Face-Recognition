
import numpy as np

from KNN import KNNClassifier


def LDA(test_Datamatrix, test_Labelmatrix, train_Datamatrix, train_Labelmatrix):
    LDA_Matrix = []
    for i in range(40):
        LDA_Matrix.append([])
    j = -1
    for i in range(200):
        if (i % 5 == 0):
            j = j + 1
        LDA_Matrix[j].append(train_Datamatrix[i])
    LDA_matrix = np.asarray(LDA_Matrix, dtype="int32")
    sb = np.zeros((10304, 10304), dtype=np.float32)
    z = np.zeros((40, 5, 10304), dtype=np.float32)
    si = np.zeros((10304, 10304), dtype=np.float32)
    mean = np.zeros((40, 10304), dtype=np.float32)
    mean = (np.mean(LDA_matrix, axis=1))
    Overall_mean = np.mean(mean, axis=0)
    for i in range(40):
        sb += (5 * (np.dot((mean[i] - Overall_mean).T, (mean[i] - Overall_mean))))
    for i in range(40):
        z[i] = (LDA_matrix[i] - mean[i])
    for i in range(40):
        si += (np.dot(z[i].T, z[i]))
    si = np.asarray(si)
    Sinv = np.linalg.inv(si)
    SinvB = np.matmul(Sinv, sb)
    eigenValues, eigenVectors = np.linalg.eig(SinvB)
    indices = eigenValues.argsort()[::-1]
    eigenValues_sorted = eigenValues[indices]
    eigenVectors_sorted = eigenVectors[:, indices]
    LDA_Train = np.dot(train_Datamatrix, eigenVectors_sorted)
    LDA_Test = np.dot(test_Datamatrix, eigenVectors_sorted)
    print("----------------------")
    # 3ayz 23ml el KNN classifier
    KNNClassifier(LDA_Train, train_Labelmatrix, LDA_Test, test_Labelmatrix)
    print("*****************************************************************************")