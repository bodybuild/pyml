'''
Created on 2016年8月18日

@author: Ma
'''



from util.datautil.DataUtil import DataUtil 
import numpy as np
from nt import listdir
from machineLearning.svm.SMOMpj import SMOMpj

filePathTrain =  "../../../data/svm/digits/trainingDigits"
filePathTest = "../../../data/svm/digits/testDigits"
trainVevctor,trainLabel = DataUtil.loadFile(filePathTrain)
testVector,testLabel = DataUtil.loadFile(filePathTest)

weigth = DataUtil.loadModel( "../../../data/svm/digits/model")



#DataUtil.storeModel(smo.weigth, "../../../data/svm/digits/model")

trainDataMat = np.mat(trainVevctor)
testDataMat = np.mat(testVector)

nagitive_train = 0.0
nagitive_test = 0.0

for i in range(len(trainLabel)):
    if trainLabel[i] != 1.0:
        nagitive_train += 1.0
        trainLabel[i] = -1.0
for i in range(len(testLabel)):
    if testLabel[i] != 1.0:
        testLabel[i] = -1.0
        nagitive_test += 1.0

error_count = 0.0
fileList = listdir(filePathTrain);

total_train = len(trainDataMat)
train_p = 0.0
train_f = 0.0
for i in range(total_train):
    predict =  np.sign(trainDataMat[i]*weigth.T)
    if predict == trainLabel[i]:
        error_count += 1.0
        if trainLabel[i] == 1.0:
            train_p += 1.0
        else:
            train_f += 1.0
    print("fileName is  %s, comeing weith classifi %f, the correct class is %f "% (fileList[i],predict,trainLabel[i]))

error_count = total_train - train_f - train_p

print("******************************************")
print(" the error_count rate  is   ", error_count/(1.0*len(trainDataMat)))
print(" the recall is %f , the precise is %f " %(train_p/(train_p + nagitive_train - train_f), train_p/(total_train - nagitive_train)))
print("******************************************")


error_count = 0.0
fileList = listdir(filePathTest);
total_test = len(testDataMat)
test_p = 0.0
test_f = 0.0

for i in range(len(testDataMat)):
    predict =  np.sign(testDataMat[i]*weigth.T)
    if predict == testLabel[i]:
        error_count += 1.0
        if testLabel[i] == 1.0:
            test_p += 1.0
        else:
            test_f += 1.0
    print("fileName is  %s, comeing weith classifi %f, the correct class is %f "% (fileList[i],predict,testLabel[i]))

error_count = total_test - test_p - test_f

print("******************************************")
print(" the error_count rate  is   ", error_count/(1.0*len(testDataMat)))
print(" the recall is %f , the precise is %f " %(test_p/(test_p + nagitive_test - test_f), test_p/(total_test - nagitive_test)))
print("******************************************")
