'''
Created on 2016年7月6日

@author: Ma
'''

import numpy as np;
from machineLearning.knn.KNN import *;
from nt import listdir
from random import random
from _ctypes import Array

class HandWrite:
    
    kn = None;
    def __init__(self):
        self.kn = KNN();
    def printTest(self):
        self.kn.printTest();
        
    def imgToVector(self,fileName):
        imgMatt = np.zeros((1,1024));
        fr = open(fileName,'r');
        lines = fr.readlines();
        numOfLines = len(lines);
        
        for i in range(numOfLines):
            for j in range(numOfLines):
                imgMatt[0,i*32 + j] = int(lines[i][j]);
        return imgMatt;
    
    def loadFile(self,filePath):
        fileList = listdir(filePath);
        numOfTraning = len(fileList);
        trainVectorMatt = np.zeros((numOfTraning,1024));
        trainLabelMatt = np.zeros(numOfTraning);
        for i in range(numOfTraning):
            labelIndex = fileList[i].find("_");
            trainLabelMatt[i] = fileList[i][0:labelIndex];
            trainVectorMatt[i,] = self.imgToVector(filePath + "/" +  fileList[i]);
        
        return trainVectorMatt,trainLabelMatt;
        
    def classifyTest(self,ratio,k):
        #ratio = 0.2 # ratio of training test
        #k = 600;
        trainVevctor,trainLabel = self.loadFile("../../../data/KNN/trainingDigits");
        testVector,testLabel = self.loadFile("../../../data/KNN/testDigits");
        
        trainNormMatt = trainVevctor;
        
        print(trainVevctor);
        print(trainNormMatt);
        
      
        testNormMatt = testVector;
        
        numOfTestDate = testNormMatt.shape[0];
       
        errorCount = 0.0;
        for i in range(numOfTestDate):
            classifyReturn = self.kn.classify(testNormMatt[i,], trainNormMatt, trainLabel, k);
         
            
            print("the handWrite came back with %d, the real anser is %d" 
                  % (classifyReturn,testLabel[i]));
            if classifyReturn != testLabel[i]:
                errorCount += 1.0;
            
        print("the total missclassifier error rate is : ", errorCount/numOfTestDate);
         
      
    def matrixTest(self):
        old_err_state = np.seterr(divide='ignore');
        a = np.matrix([[2,4],[6,8]]);
        print(a);
        b = np.matrix([[2,2],[3,0]]);
        print(b);
        print(np.divide(a,b));
    
    
hw = HandWrite();
#hw.matrixTest();
hw.classifyTest(1, 5);
