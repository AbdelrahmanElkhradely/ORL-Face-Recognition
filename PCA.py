import numpy as np


def PCA (test_Datamatrix, test_Labelmatrix, train_Datamatrix, train_Labelmatrix):
    mean = np.mean(train_Datamatrix, axis=0)
    z = train_Datamatrix - mean
    z_test = test_Datamatrix - np.mean(test_Datamatrix, axis=0)
    cov_matrix = np.cov(z, rowvar=0, bias=1)
    eigenValues, eigenVectors = np.linalg.eig(cov_matrix)
    indices = eigenValues.argsort()[::-1]
    eigenValues_sorted = eigenValues[indices]
    eigenVectors_sorted = eigenVectors[:, indices]
    r_values = [0.8, 0.85, 0.9, 0.95]
    for r in r_values:
        for i in range(r, 10304):
            B = float(sum(eigenValues_sorted))
            T = float(sum(eigenValues_sorted[:i]))
            if (T / B >= r):
                alpha=i
                break
        new_matrix = eigenVectors_sorted[:, 0: alpha + 1]
        pca_Train = np.dot(new_matrix.T, z.T)
        pca_Test = np.dot(new_matrix.T, z_test.T)
        print("R =" + str(r))
        print("----------------------")
        # 3ayz 23ml el KNN classifier
        print("*****************************************************************************")
