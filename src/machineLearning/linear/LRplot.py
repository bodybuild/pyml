'''
Created on 2016年8月30日

@author: Ma
'''


import matplotlib.pyplot as plt
import numpy as np
from util.datautil.DataUtil import DataUtil
from numpy.linalg import linalg
import time
class LRplot:
    
    def LRStand(self,dataIn,classLabel):
        dataIn = np.mat(dataIn)
        classLabel = np.mat(classLabel).T
        xInverse = dataIn.T*dataIn
       
        if linalg.det(xInverse) == 0.0:
            print(" matrix of X.T*X  is singular,cannot do inverse")
            return 0
        else:
            w = linalg.solve(xInverse, dataIn.T*classLabel)
            return w
    def ridgeRegress(self,dataIn,classLabel,lam = 0.2):
        dataIn = np.mat(dataIn)
        classLabel = np.mat(classLabel).T
        xInverse = dataIn.T*dataIn + np.eye(np.shape(dataIn)[1])*lam
        
        w = linalg.solve(xInverse, dataIn.T*classLabel)
    
        return w
    
    def LinearRegression(self):
        filePath1 = "../../../data/LR/ex0.txt"
        filePath2 = "../../../data/LR/ex1.txt"
        dataIn1,classLabel1 = DataUtil.loadDateFloat(filePath1)
        dataIn2,classLabel2 = DataUtil.loadDateFloat(filePath2)
        
        w1 = self.LRStand(dataIn1, classLabel1)
        yMat1 = dataIn1*w1
        
        w2 = self.LRStand(dataIn2, classLabel2)
        yMat2 = dataIn2*w2
        
        
        plt.figure(1)
        plt.clf()
        plt.subplot(211)
        plt.scatter(dataIn1[:,1],classLabel1)
        plt.plot(dataIn1[:,1],yMat1, c = 'b')
        plt.subplot(212)
        plt.scatter(dataIn2[:,1],classLabel2)
        plt.plot(dataIn2[:,1],yMat2, c = 'b')
       
        
        
        print(np.corrcoef(yMat1.T,np.mat(classLabel1)))
        print(np.corrcoef(yMat2.T,np.mat(classLabel2)))
        
        plt.show()
        
    def LWRStand(self,X,dataIn,classLabel, m = 1.0):
        dataIn = np.mat(dataIn)
        classLabel = np.mat(classLabel).T
        lines,cols = np.shape(dataIn)
        K = np.mat(np.eye((lines)))
        for i in range(lines):
            diff = (dataIn[i] - X)*(dataIn[i] - X).T
            
            K[i,i] = np.exp(diff/((-1)*m**2))
      
        xInverse = dataIn.T*(K*dataIn)
       
        if linalg.det(xInverse) == 0.0:
            print(" matrix of X.T*X  is singular,cannot do inverse")
            return 0
        else:
            w = linalg.solve(xInverse, dataIn.T*K*classLabel)
            
            return X*w
        
    def LWR(self):
        filePath1 = "../../../data/LR/ex0.txt"
        filePath2 = "../../../data/LR/ex1.txt"
        dataIn1,classLabel1 = DataUtil.loadDateFloat(filePath2)
        #dataIn2,classLabel2 = DataUtil.loadDateFloat(filePath2)
       
        lines,cols = np.shape(dataIn1)
       
        yMat_LWR = np.zeros((lines,1))
       
        w1 = self.LRStand(dataIn1, classLabel1)
        yMat_LR = dataIn1*w1  
        
        plt.figure(1)
        plt.clf()
        plt.subplot(211)
        plt.scatter(dataIn1[:,1],classLabel1)
        plt.plot(dataIn1[:,1],yMat_LR, c = 'b')
        plt.subplot(212)
        
        for i in range(lines):
            yMat_LWR[i] = self.LWRStand(dataIn1[i], dataIn1, classLabel1,m=0.005)
            print(" compute LWR i = ",i)
        
        xCopy = np.copy(dataIn1[:,1])
        sortIndex = np.argsort(xCopy)
        
        plt.scatter(dataIn1[:,1],classLabel1)
        plt.plot(xCopy[sortIndex],yMat_LWR[sortIndex], c = 'b')
        '''
        plt.subplot(513)
        
        for i in range(lines):
            yMat_LWR[i] = self.LWRStand(dataIn1[i], dataIn1, classLabel1,m=0.5)
            print(" compute LWR i = ",i)
        
        xCopy = np.copy(dataIn1[:,1])
        sortIndex = np.argsort(xCopy)
        
        plt.scatter(dataIn1[:,1],classLabel1)
        plt.plot(xCopy[sortIndex],yMat_LWR[sortIndex], c = 'b')
        
        plt.subplot(514)
        
        for i in range(lines):
            yMat_LWR[i] = self.LWRStand(dataIn1[i], dataIn1, classLabel1,m=0.1)
            print(" compute LWR i = ",i)
        
        xCopy = np.copy(dataIn1[:,1])
        sortIndex = np.argsort(xCopy)
        
        plt.scatter(dataIn1[:,1],classLabel1)
        plt.plot(xCopy[sortIndex],yMat_LWR[sortIndex], c = 'b')
        
        plt.subplot(515)
        
        for i in range(lines):
            yMat_LWR[i] = self.LWRStand(dataIn1[i], dataIn1, classLabel1,m=0.01)
            print(" compute LWR i = ",i)
        
        xCopy = np.copy(dataIn1[:,1])
        sortIndex = np.argsort(xCopy)
        
        plt.scatter(dataIn1[:,1],classLabel1)
        plt.plot(xCopy[sortIndex],yMat_LWR[sortIndex], c = 'b')
        '''
        plt.show()
    
    
    def stageWise(self,dataIn,classLabel,lam = 0.01,maxInter = 50):
       
        dataIn = np.mat(dataIn)
        classLabel = np.mat(classLabel).T
        yMean = np.mean(classLabel,0)
        classLabel = classLabel - yMean    
        dataIn = self.regularize(dataIn)
        lines,cols = np.shape(dataIn)
       
        w = np.zeros((cols,1))
        wChage = w.copy()
        wBest = w.copy()
        wReturn = np.zeros((maxInter,cols))
        for m in range(maxInter):
            minError = np.Inf
            for i in range(cols):
                for labelValue in [-1,1]:
                    print(labelValue)
                    wChage = w.copy()
                    print("   wChage is ", wChage.T)
                    wChage[i] += labelValue*lam
                    
                    yMat = dataIn*wChage
                   
                    error = self.rssError(classLabel.A,yMat.A)
                    print(error)
                   
                    if error < minError:
                        wBest = wChage
                        minError = error
                    print("   w is ",w.T)
                    print("   the feature is %d, sign is %d, minError is %f ."%(i,labelValue,minError))
            w = wBest.copy()
            print(w.T)
            wReturn[m] = w.T
                       
               
            print("this is %d's iterator, error is %f "%(m,minError))
        return wReturn
       
        
    def regularize(self,xMat):#regularize by columns
        inMat = xMat.copy()
        inMeans = np.mean(inMat,0)   #calc mean then subtract it off
        inVar = np.var(inMat,0)      #calc variance of Xi then divide by it
        inMat = (inMat - inMeans)/inVar
        return inMat
    def rssError(self,yArr,yHatArr): #yArr and yHatArr both need to be arrays
        return ((yArr-yHatArr)**2).sum()
    def compareModel(self):
        filePath2 = "../../../data/LR/abalone.txt"
        dataIn,classLabel = DataUtil.loadDateFloat(filePath2)
        
        wReturn = self.stageWise(dataIn, classLabel, maxInter = 500)
        print(wReturn)
        '''
        
        numOfplt = 30
        wMat = np.zeros((numOfplt,np.shape(dataIn)[1])) 
        for i in range(numOfplt):
            wRR = self.ridgeRegress(dataIn, classLabel, lam = np.exp(i - 10))
            wMat[i] = wRR.T
        '''
        plt.figure(1)
        plt.clf()
        plt.xlabel("IterNum")
        plt.plot(wReturn)
       
        plt.savefig("../../../img/LR/linearRegress_%s.png"%(time.time()))
        plt.show()
        
    def _run(self):
        #self.LinearRegression()
        self.LWR()
        #self.compareModel()

lr = LRplot()
lr._run()