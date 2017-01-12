'''
Created on 2016年9月6日

@author: Ma
'''

import numpy as np
import matplotlib.pyplot as plt
from util.datautil.DataUtil import DataUtil 
import time
class Kmeans:
    
    def calDistance(self,vecA,vecB):
        return np.sum(np.power(vecA - vecB,2))
    
    def createCenter(self,dataIn,k):
        lines,cols = np.shape(dataIn)
        centers = np.mat( np.zeros((k,cols)))
        for i in range(cols):
            minCols = np.min(dataIn[:,i])
            maxCols = np.max(dataIn[:,i])
            centers[:,i] = minCols + (maxCols - minCols)*np.random.rand(k,1)
    
        return centers
    def Kmeans(self,dataIn,k = 3,maxIter=100):
        
        lines,cols = np.shape(dataIn)
        centers = self.createCenter(dataIn, k)
        centerChange = True
        iterNum = 0 
        numSplit = np.mat(np.zeros((lines,2)))
        while centerChange and iterNum < maxIter :
            centerChange = False
            for i in range(lines):
                bestClusterIndex = -1;bestClusterDis = np.Inf
                for j in range(k):
                    distance = self.calDistance(dataIn[i], centers[j])
                    
                    if distance < bestClusterDis:
                        bestClusterIndex = j
                        bestClusterDis = distance
                if numSplit[i,0] != bestClusterIndex:
                    centerChange = True
                numSplit[i] = bestClusterIndex,bestClusterDis
          
            for j in range(k):
                dataSplit = dataIn[np.nonzero(numSplit[:,0] == j)[0]]
            
                centers[j,:] = np.mean(dataSplit,axis=0)
            iterNum +=1
        return centers,numSplit
            
                
    def trainKmeans(self,filePath):
        dataIn = np.mat( DataUtil.loadDateMatt(filePath ))
        centers = self.createCenter(dataIn,6)
        print(centers)
        fig = plt.figure()
        fig.clf()
        plt.scatter(dataIn[:,0],dataIn[:,1],marker='o',c='b')
        plt.scatter(centers[:,0],centers[:,1],marker='x',c='r')
        centers,splitData =  self.Kmeans(dataIn,k=6)
        print(centers)
        plt.scatter(centers[:,0],centers[:,1],marker='x',c='green')
        plt.savefig("../../../img/kmeans/kmeans_%s.png"%(time.time()))
        plt.show()
        
        
        
    
    def binaryKmeans(self,dataIn,k=3):
       
        lines,cols = np.shape(dataIn)
        centerInit = np.mean(dataIn,axis=0)  
        centerList = [centerInit.tolist()[0]]
        numSplit = np.mat(np.zeros((lines,2)))
        for i in range(lines):
            numSplit[i:,] = 0,self.calDistance(dataIn[i,:], centerList[0])
        
        while len(centerList) < k:
            bestSSE = np.Inf
            for i in range(len(centerList)):
                dataSplit = dataIn[np.nonzero( numSplit[:,0] == i)[0]].copy()
                centerTmp,dataSplitTmp =  self.Kmeans(dataSplit, 2)
                SSErrorSplit = np.sum(dataSplitTmp[:,1])
                SSErrorOther = np.sum(numSplit[np.nonzero(numSplit[:,0] != i)[0],1])
                SSError = SSErrorSplit + SSErrorOther
                
                print("  split %d,  SSError  = %f ,bestSSE = %f "%(i,SSError,bestSSE)   )
                
                
                if SSError < bestSSE:
                    bestSplitIndex = i
                    bestSSE = SSError
                    bestNewCenter = centerTmp
                    bestDataSplit = dataSplitTmp
                    bestData  = dataSplit
            print("bestSplitIndex  = " ,bestSplitIndex) 
            print("bestSSE  = " ,bestSSE)       
            print(bestNewCenter)
            data0 = np.nonzero(bestDataSplit[:,0] == 0)[0]
            data1 = np.nonzero(bestDataSplit[:,0] == 1)[0]
                     
            bestDataSplit[data0,0] = bestSplitIndex  
            bestDataSplit[data1,0] = len(centerList)

            numSplit[np.nonzero(numSplit[:,0] == bestSplitIndex)[0]] = bestDataSplit 
           
            centerList[bestSplitIndex] = bestNewCenter[0,:].tolist()[0]
            centerList.append(bestNewCenter[1,:].tolist()[0])
        return np.mat(centerList),numSplit    
    
    def _run(self):
        filePath = "../../../data/kmeans/testSet.txt"
        dataIn = np.mat( DataUtil.loadDateMatt(filePath ))
        #self.trainKmeans(filePath)=====
        
        k  = 6
        fig = plt.figure()
        fig.clf()
        plt.scatter(dataIn[:,0],dataIn[:,1],marker='o',c='blue')
        
        centers = self.createCenter(dataIn,k)
        plt.scatter(centers[:,0],centers[:,1],marker='x',c='grey',s=40)
        
        centers,splitData =  self.Kmeans(dataIn,k)
        plt.scatter(centers[:,0],centers[:,1],marker='x',c='green',s=40)
        
        centers,splitData = self.binaryKmeans(dataIn, k)
        plt.scatter(centers[:,0],centers[:,1],marker='x',c='r',s=40)
        
        plt.savefig("../../../img/kmeans/kmeans_%s.png"%(time.time()))
        plt.show()
        
    

kmeans = Kmeans()
kmeans._run()