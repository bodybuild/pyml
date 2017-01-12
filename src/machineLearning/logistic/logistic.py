'''
Created on 2016年7月19日

@author: Ma
'''
import numpy as np;
from numpy import size
class Logistic:
    @staticmethod
    def loadDate(path):
        
        fileMatt = np.loadtxt(path,dtype = np.float);
      
        return fileMatt[:,0:-1],fileMatt[:,-1];
        """
        docList = [];
        for line in lines:
            doc = re.split("[\t,\n]", line)[0:-1];
            docList.append(doc);
        docMatt = np.mat(docList);
        return docMatt[:,0:-1],docMatt[:,-1];
        """
    @staticmethod
    def sigmod(xVec):
        return 1.0/(1 + np.exp(-xVec));
    
    def batchGradientDescent(self,path,threshold,alpha,numIter = 1000):
        dataVec,labelVec =  self.loadDate(path);
        
        lines = len(dataVec);
        
        x1 = np.ones((lines,1)); #增加 截距
        
        dataMat = np.mat(dataVec, dtype = np.float);
      
        dataMat = np.column_stack((x1,dataMat));
        lines,cols = np.shape(dataMat);
        weights = np.zeros((cols,1)) ;

        labelMat = np.transpose( np.mat(labelVec));
        
        weigthX = [];
        weigthY = [];
        for i in range(numIter):
            
            ww = dataMat*weights;
            hx = self.sigmod(ww);
            error = (labelMat - hx);
            weights = weights + alpha*dataMat.transpose()*error;
            weigthX.append(i);
            weigthY.append(list(np.asarray(weights).reshape(-1)));
        return weights,weigthX,weigthY;
      
   
        
    def stochasticGradientDescent(self,path,threshold,alpha,numIter = 1000):
        dataVec,labelVec =  self.loadDate(path);
        lines = len(dataVec);
        
        x1 = np.ones((lines,1)); #增加 截距
        dataMat = np.mat(dataVec, dtype = np.float);
        dataMat = np.column_stack((x1,dataMat));
        
        
        lines,cols = np.shape(dataMat);
        weights = np.zeros((cols,1)) ;
        
        weigthX = [];
        weigthY = [];

        for i in range(lines):
            hx = self.sigmod(dataMat[i]*weights);
            error = (labelVec[i] - hx);
            weights = weights + alpha*dataMat[i].transpose()*error;
            weigthX.append(i);
            weigthY.append(list(np.asarray(weights).reshape(-1)));
     
        return weights,weigthX, weigthY;
    
    def improveSGD(self,path,threshold,alpha,numIter = 1000,randomFlag = False,alphaFlag=False):
        
        dataVec,labelVec =  self.loadDate(path);
        lines = len(dataVec);
        
        x1 = np.ones((lines,1)); #增加 截距
        dataMat = np.mat(dataVec, dtype = np.float);
        dataMat = np.column_stack((x1,dataMat));
        
        
        lines,cols = np.shape(dataMat);
        weights = np.zeros((cols,1)) ;
        
        weigthX = [];
        weigthY = [];
        numOfIter = 0;
        for j in range(numIter):
            for i in range(lines):
                if alphaFlag :
                    alpha =  alpha*(1 - (i+j+1.0)/(numIter*lines));
                numOfIter += 1;
                if randomFlag :
                    index = np.random.randint(0,lines);
                else:
                    index = i;
                hx = self.sigmod(dataMat[i]*weights);
                error = (float(labelVec[index]) - hx);
                weights = weights + alpha*dataMat[index].transpose()*error;
                weigthX.append(numOfIter);
                weigthY.append(list(np.asarray(weights).reshape(-1)));
     
        return weights,weigthX, weigthY;
        
        
        
        
    def _run(self):
        maxCycle = 1000;
        threshold = 0.01;
        alpha = 0.001
        path = "../../../data/logistic/testSet.txt";
        #print(self.batchGradientDescent(threshold,alpha,numIter = 1000));
        print(self.stochasticGradientDescent(path, threshold, alpha,numIter = 1000))
        """
        nx = np.array([[1,2,3],[4,5,6],[7,8,9]]);
        cols = nx[:,-1];
        nx = np.column_stack((nx,cols));
        print(nx);
        """

#lg = Logistic();
#lg._run();
