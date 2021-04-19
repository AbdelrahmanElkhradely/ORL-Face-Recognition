def split(dataset,labels):
    test_Datamatrix = []
    test_Labelmatrix = []
    train_Datamatrix = []
    train_Labelmatrix = []
    i=0
    for i in range(400):
        if (i % 2 == 1):
            test_Datamatrix.append(dataset[i])
            test_Labelmatrix.append(labels[i])
        else:
            train_Datamatrix.append(dataset[i])
            train_Labelmatrix.append(labels[i])
    return test_Datamatrix,test_Labelmatrix,train_Datamatrix,train_Labelmatrix