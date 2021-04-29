def split(dataset,labels):
    test_Datamatrix = []
    test_Labelmatrix = []
    train_Datamatrix = []
    train_Labelmatrix = []
    from sklearn.model_selection import train_test_split
    train_Datamatrix, test_Datamatrix, train_Labelmatrix, test_Labelmatrix = train_test_split(
        dataset, labels, test_size=0.3, random_state=42)
    return test_Datamatrix,test_Labelmatrix,train_Datamatrix,train_Labelmatrix