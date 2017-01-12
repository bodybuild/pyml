'''
Created on 2016年9月28日

@author: Ma
'''


import numpy as np 
import numpy.linalg as la
class SVD:
    def loadTxt(self,filePath):
        fr = open(filePath,"r")
        lines = fr.readlines()
        lineArr = []
        for i in range(len(lines)):
            lineArr.append([float(s)  for s in lines[i].strip()])
        return np.mat(lineArr)
    
    def printMat(self,dataMatt,thresh):
        lines,cols = np.shape(dataMatt)
        for i in range(lines):
            data = []
            for j in range(cols):
                if dataMatt[i,j] < thresh:
                    print('0',end=' ')
                else:
                    print('1',end=' ')
            print("")
   
    def imgCompress(self,numSV=3, thresh=0.8):
        myl = []
        filePath = "../../../data/SVD/0_5.txt"
        for line in open(filePath).readlines():
            newRow = []
            for i in range(32):
                newRow.append(int(line[i]))
            myl.append(newRow)
        myMat = np.mat(myl)
        self.printMat(myMat, thresh)
        U,Sigma,VT = la.svd(myMat)
        SigRecon = np.mat(np.zeros((numSV, numSV)))
        for k in range(numSV):#construct diagonal matrix from vector
            SigRecon[k,k] = Sigma[k]
        print(SigRecon)
        reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
        self.printMat(reconMat, thresh)
    
    
    def _run(self,k=3,thresh=0.8):
        filePath = "../../../data/SVD/0_5.txt"
       
        dataMatt = self.loadTxt(filePath)
        print(np.shape(dataMatt))
        U,Sigma,VT = la.svd(dataMatt)
        sigmaMat = Sigma[:k]*np.eye(k,k)
        
        reMatt = U[:,:k]*sigmaMat*VT[:k,:]
       
        print(sigmaMat)
        self.printMat(dataMatt, thresh)
        print("*************************************")
        self.printMat(reMatt, thresh)

svd = SVD()
svd._run()
svd.imgCompress()

