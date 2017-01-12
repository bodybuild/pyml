'''
Created on 2016年9月22日

@author: Ma
'''

import numpy as np 
from util.datautil.DataUtil import DataUtil
class PCA:
    
    def loadDataSet(self,fileName, delim='\t'):
        fr = open(fileName)
        stringArr = [line.strip().split(delim) for line in fr.readlines()]
        print(type(stringArr[0][0]))
        return np.mat(stringArr)
    
    def pca(self,dataMat, topNfeat=9999999):
      
        meanVals = np.mean(dataMat, axis=0)
        print(meanVals)
        print(dataMat)
        meanRemoved = dataMat - meanVals #remove mean
        covMat = np.cov(meanRemoved, rowvar=0)
        print(covMat)
        eigVals,eigVects = np.linalg.eig(np.mat(covMat))
        print(eigVals)
        print("eigVals = ", eigVects)
        eigValInd = np.argsort(eigVals)            #sort, sort goes smallest to largest
        eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
        redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
        lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
        print(lowDDataMat)
        print(redEigVects.T)
        reconMat = (lowDDataMat * redEigVects.T) + meanVals
        print(reconMat)
        return lowDDataMat, reconMat
 
    def _run(self):
        filePath = "../../../data/PCA/testSet3.txt"
        
        dataMat = DataUtil.loadDateMatt(filePath)
     
        a,b =  self.pca(dataMat, 1)
      
pca = PCA()
pca._run()