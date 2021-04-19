import Load_DataSet
import SplitDataSet
if __name__ == '__main__':
    images,labels=Load_DataSet.get_dataset()
    test_Datamatrix, test_Labelmatrix, train_Datamatrix, train_Labelmatrix= SplitDataSet.split(images,labels)


