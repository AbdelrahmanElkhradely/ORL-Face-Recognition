import numpy as np

from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
def KNNClassifier(train_data,train_label,test_data,test_label):
    neighbours = [1,3,5,7]
    accuracy_matrix = []
    print("train : ")
    print(len(train_data) )
    print("test : ")
    print(len(test_data) )

    for i,neighbour in zip(range(len(neighbours)),neighbours):
        print("When Neighbour = " + str(neighbour) )
        clf = KNeighborsClassifier(n_neighbors = neighbour, weights = 'distance')
        clf.fit(train_data.T, train_label)
        output = clf.predict(test_data.T)
        accuracy_matrix.append(accuracy_score(output,test_label))
        print("accuracy score = " + str(accuracy_matrix[i]))
        count = 0
        for i in range(len(output)):
            if((output[i]) != (test_label[i])):
                print( str(i) + "-" + "classified as: " + str(output[i]) + " actual is: " + str(test_label[i]) + "     Misclassified")
                count+=1
            else:
                print(str(i) + "-" + "classified as: " + str(output[i]) + " actual is: " + str(test_label[i]) )
        print("Number of Misclassified is " + str(count))
        print("=========================================")
    plt.plot(accuracy_matrix, neighbours)
    plt.show()
