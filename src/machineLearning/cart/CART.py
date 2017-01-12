'''
Created on 2016年9月5日

@author: Ma
'''

import numpy as np
import matplotlib.pyplot as plt
import time
from util.datautil.DataUtil import DataUtil
from machineLearning.linear.LRplot import LRplot 
class CART:
    def computeLeafNode(self,dataIn):
        return np.mean(dataIn[:,-1])
    def computeLeafNodeLR(self,dataIn):
        lr = LRplot()
        ws = lr.ridgeRegress(dataIn[:,0:-1], dataIn[:,-1].T)
        return ws
    def splitData(self,dataIn,splitIndex,splitValue):
        mat1 = dataIn[np.nonzero(dataIn[:,splitIndex] < splitValue)[0]]
        mat2 = dataIn[np.nonzero(dataIn[:,splitIndex] >= splitValue)[0]]
        return mat1,mat2
    def computeLossCART(self,dataIn):
        return np.var(dataIn[:,-1])*len(dataIn[:,-1])
    
    def computeLossLR(self,dataIn):
        lr = LRplot()
        ws = lr.ridgeRegress(dataIn[:,0:-1], dataIn[:,-1].T)
        diff = dataIn[:,-1] -  dataIn[:,0:-1]*ws 
        return np.multiply(diff,diff).sum()
    def chooseBestSplit(self,dataIn,errType,leafType,ops={"errorThreshold":1,"minSample":4}):

        if len(set(dataIn[:,-1].T.tolist()[0])) == 1:
            return None,leafType(dataIn)
        lines,cols = np.shape(dataIn)
        totalE = errType(dataIn)
        bestIndex = 0;bestValue = 0;bestE = np.Inf
        for col in range(cols - 1):
            for splitValue in dataIn[:,col]:
                print(splitValue)
                mat1,mat2 = self.splitData(dataIn, col, splitValue)
                if np.shape(mat1)[0] < ops["minSample"] or np.shape(mat2)[0] < ops['minSample']:
                    continue
                splitError = errType(mat1) + errType(mat2)
                print("cols = %d, splitValue = %f,splitError = %f, bestE = %f ,bestCos = %d,bestValue = %f "%(col,splitValue, splitError,bestE,bestIndex,bestValue))
               
                if splitError < bestE:
                    bestE = splitError
                    bestIndex = col
                    bestValue = splitValue
        if (totalE - bestE) < ops['errorThreshold'] : # if error has not enough change (pre-pruning)
            return None,leafType(dataIn)
        return bestIndex,bestValue
    def createCARTTree(self,dataIn,errType,leafType,ops={"errorThreshold":1,"minSample":4}):
        bestIndex,bestValue = self.chooseBestSplit(dataIn, errType, leafType, ops)
        if bestIndex == None:
            return bestValue
        dataMat1,dataMat2 = self.splitData(dataIn, bestIndex, bestValue)
        cartTree = {}
        cartTree["splitIndex"] = bestIndex
        cartTree["splitValue"] = bestValue.tolist()[0][0]
        cartTree["left"] = self.createCARTTree(dataMat1, errType, leafType, ops)
        cartTree['right'] = self.createCARTTree(dataMat2, errType, leafType, ops)
        return cartTree
    
    def isTree(self,tree):
        return (type(tree).__name__ == 'dict')
    
    def mergePreValueLR(self,testData,tree):
        
        lmat,rmat = self.splitData(testData, tree['splitIndex'], tree['splitValue'])
           
        diffL = lmat[:,-1] - lmat[:,0:-1]*tree['left']
        diffR = rmat[:,-1] - rmat[:,0:-1]*tree['right']
        errorNoMerge = np.multiply(diffL,diffL).sum() + np.multiply(diffR,diffR).sum()
        ws = self.computeLeafNodeLR(testData)
        noMergeMean = testData[:,0:-1]*ws
        errorMerge = np.multiply(testData[:,-1] - noMergeMean,testData[:,-1] - noMergeMean).sum()
        if errorMerge < errorNoMerge:
            print("pruning")
            return ws
        else:
            return tree
        
    def mergePreValue(self,testData,tree):
        lmat,rmat = self.splitData(testData, tree['splitIndex'], tree['splitValue'])
           
        diffL = lmat[:,-1] - tree['left']
        diffR = rmat[:,-1] - tree['right']
        errorNoMerge = np.multiply(diffL,diffL).sum() + np.multiply(diffR,diffR).sum()
        noMergeMean = (tree['left'] + tree['right']) /2.0
        errorMerge = np.multiply(testData[:,-1] - noMergeMean,testData[:,-1] - noMergeMean).sum()
        if errorMerge < errorNoMerge:
            print("pruning")
            return noMergeMean
        else:
            return tree
    
    def pruning(self,tree,testData,mergeType):
        if self.isTree(tree['left']) or self.isTree(tree['right']):
            lmat,rmat = self.splitData(testData, tree['splitIndex'], tree['splitValue'])
            if self.isTree(tree['left']):
                tree['left'] = self.pruning(tree['left'], lmat,mergeType)
            if self.isTree(tree['right']):
                tree['right'] = self.pruning(tree['right'], rmat,mergeType)
        else:
            return mergeType(testData,tree)
        return tree    
    
    
    def trainModel(self,trainPath,testPath):
        
        dataIn = np.mat(DataUtil.loadDateMatt(trainPath))
        
        tree = self.createCARTTree(dataIn, self.computeLossCART, self.computeLeafNode, ops={"errorThreshold":1,"minSample":4})
        print(tree)
        
        testData =  np.mat(DataUtil.loadDateMatt(testPath))
        tree = self.pruning(tree, testData,mergeType=self.mergePreValue)
        print(tree)
        
        DataUtil.storeModel(tree, "../../../data/CART/model-1")
    
    
    def trainModelLR(self,trainPath,testPath):
        
        dataIn = np.mat(DataUtil.loadDateMatt(trainPath))
        
        tree = self.createCARTTree(dataIn, errType=self.computeLossLR, leafType=self.computeLeafNodeLR)
        print(tree)
       
        testData =  np.mat(DataUtil.loadDateMatt(testPath))
        tree = self.pruning(tree, testData, mergeType=self.mergePreValueLR)
        print(tree)
        DataUtil.storeModel(tree, "../../../data/CART/model-LR")
        
    def predict(self,X,tree):
        return tree
    def predictLR(self,X,tree):
        return X*tree
    def looptree(self,X,tree,preType):
     
        if self.isTree(tree):
           
            if X[tree['splitIndex']] < tree['splitValue']:
                return self.looptree(X, tree['left'],preType)
            else:
                return self.looptree(X, tree['right'],preType)
        else:
            return preType(X,tree)
    def testModel(self,testPath):
        testData,classLabel =  DataUtil.loadDateFloat(testPath)
     
        
  
        lines,cols = np.shape(testData)
       
        xCopy = testData[:,0].copy()
        yCopy = classLabel.copy()
        xSortIndex = np.argsort(xCopy, axis=0)
    
        fig =  plt.figure(1)
        fig.clf()
        print("")
        plt.scatter(xCopy[xSortIndex],yCopy[xSortIndex])
       
        plt.plot(xCopy[xSortIndex],yCopy[xSortIndex], c = 'b')
        
        yMat = np.zeros((lines,1))
        tree = DataUtil.loadModel("../../../data/CART/model-1")
        for i in range(lines):
            yMat[i] = self.looptree(testData[i], tree,preType=self.predict)
        
        plt.plot(xCopy[xSortIndex],yMat[xSortIndex], c = 'red')
        
        yMat = np.zeros((lines,1))
        tree = DataUtil.loadModel("../../../data/CART/model-LR")
        for i in range(lines):
            yMat[i] = self.looptree(testData[i], tree,preType=self.predictLR)
        
            
        plt.plot(xCopy[xSortIndex],yMat[xSortIndex], c = 'green')
        plt.xlabel("The green line, lack intercept ,so when x = 0,predictY =0, In other words, startPoint is (0,0)")
        
        plt.savefig("../../../img/CART/CART_%s.png"%(time.time()))
        plt.show()
        
    def _run(self):
        trainPath = "../../../data/CART/bikeSpeedVsIq_train.txt"
        
        testPath = "../../../data/CART/bikeSpeedVsIq_test.txt"
       
        self.trainModel(trainPath,testPath)
        self.trainModelLR(trainPath,testPath)
        self.testModel(testPath)
       
        '''
        fig =  plt.figure(1)
        fig.clf()
        plt.scatter(dataIn[:,1],classLabel)
     
        plt.savefig("../../../img/CART/CART_%s.png"%(time.time()))
        plt.show()
        '''

cart = CART()
cart._run()