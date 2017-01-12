'''
Created on 2016年8月21日

@author: Ma
'''

import numpy as np
from util.datautil.DataUtil import DataUtil
from math import inf
from pyexpat import model
import matplotlib.pyplot as plt

class AdaBoosting:

    def __init__(self):
        print("initial")
        
    def treeClassify(self,dataIn,dimen,inequal,thresh):
        reArray = np.ones((self.lines,1))
        if inequal == 'lt':
            reArray[dataIn[:,dimen] < thresh ]  = -1.0
        elif inequal == 'gt':
            reArray[dataIn[:,dimen] >= thresh] = -1.0 

        return reArray
    def builtTree(self,dataIn,classLabel):
     
        numOfstep = 10
        bestTree = {}
        minError = inf
        bestClass = np.mat(np.zeros((self.lines,1)))
        
        for i in range(self.cols):
            rangeMin = dataIn[:,i].min()
            rangeMax = dataIn[:,i].max()
            stepSize = (rangeMax - rangeMin)/numOfstep
            for j in range(-1,int(numOfstep) + 1):
                for inequal in ['lt','gt']:
                    threshValue = rangeMin + (float(stepSize)*j )
                    predictVals = self.treeClassify(dataIn, i, inequal, threshValue)
                    
                    errArray = np.ones((self.lines,1))
                    errArray[predictVals == classLabel] = 0
                    weigthError = self.D.T*errArray
                    
                   
                    if(weigthError < minError):
                        bestTree['dimen'] = i
                        bestTree['thresh'] = threshValue
                        bestTree['inequal'] = inequal
                        minError = weigthError
                        bestClass = predictVals.copy()
                        
        #print("  Tree split dimen = %d, thresh = %.2f, thresh inequal = %s,  the weightErr = %f" \
        #%(bestTree['dimen'],bestTree['thresh'], bestTree['inequal'],minError))
                    
        return bestTree,minError,bestClass
            
    def boosting(self,dataIn,classLabel,maxNum = 100):
        weekClassiy = []
        addClass = np.zeros((self.lines,1))
        for i in range(min(maxNum,self.lines)):
            
            
            
            bestTree,minError,bestClass = self.builtTree(dataIn, classLabel)
            alpha = 0.5*np.log((1.0 - minError)/max(minError,1e-16))
            bestTree['alpha'] = alpha.A[0][0]
          
            expon =  np.multiply(-1*bestTree['alpha']*classLabel,bestClass)
            updataD= np.multiply(self.D,np.exp(expon))
            self.D = updataD/updataD.sum()
            
            weekClassiy.append(bestTree)
            
            addClass += np.multiply(bestTree['alpha'],bestClass)
            
          
            
          
            errMat =  np.nonzero( np.sign(addClass) + classLabel)[0]
          
            addError = 1.0 - len(errMat)/self.lines
           
            # print("  with %d th weekClassify ,the last error is %f "%(i,addError))
            
            if addError == 0:
                print("  no misclassify ")
                break;
            
        return weekClassiy ,addError
    
    def trainModel(self):
        filePath = "../../../data/boosting/horseColicTraining2.txt"
        trainVector,classLabel = DataUtil.loadDateFloat(filePath)
       
        '''
        trainVector = np.matrix([[ 1. ,  2.1],
            [ 2. ,  1.1],
            [ 1.3,  1. ],
            [ 1. ,  1. ],
            [ 2. ,  1. ]])
        classLabel = [1.0, 1.0, -1.0, -1.0, 1.0]
        '''
        dataIn = np.mat(trainVector)
        classLabel = np.mat(classLabel).T
        self.lines,self.cols = np.shape(dataIn)
        
        maxNum = [1,10,50,100,500,1000]
        for i in maxNum:
            self.D = np.mat(np.ones((self.lines,1))/self.lines)
            bestTree,trainError = self.boosting(dataIn, classLabel,maxNum=i)
            model = {}
            model['tree'] = bestTree
            model['trainError'] = trainError
            DataUtil.storeModel(model, ("../../../data/boosting/model-%d"%(i)))
            print("the boosting week classify  = " ,model)
    
    def predictBoosting(self,dataIn,classLabel,model):  
    
        addClass = np.zeros((self.lines,1))
        for dictModel in model:
           
            bestClass =  self.treeClassify(dataIn, dictModel['dimen'], dictModel['inequal'], dictModel['thresh'])
            addClass += np.multiply(dictModel['alpha'],bestClass)
            
       
        return addClass
    def predict(self):
        filePath = "../../../data/boosting/horseColicTest2.txt"
        trainVector,classLabel = DataUtil.loadDateFloat(filePath) 
        dataIn = np.mat(trainVector)
        classLabel = np.mat(classLabel).T
        self.lines,self.cols = np.shape(dataIn)
        
        '''
        model = DataUtil.loadModel("../../../data/boosting/model-100")
            
        print(" the training Error of this model : ", model['trainError'])
        scores  = self.predictBoosting(dataIn, classLabel, model['tree'])
        predict  = np.sign(scores)
        errMat =  np.nonzero(predict + classLabel)[0]
        addError = 1.0 - len(errMat)/self.lines
        print(" the test Error of this model : ", addError)
        
        scoreArray = np.argsort(-1*scores.T).reshape(-1)
  
        
        
        numOfPos = np.sum( classLabel == 1.0 )
        numOfNag = np.sum( classLabel == -1.0 )
        
        yStep = float(1.0/numOfPos)
        xStep = float(1.0/numOfNag)
        
        fig = plt.figure(1)
        fig.clf()
     
        startPoint = [0.0,0.0]
        sumOfPos = 0.0
        sumOfNag = numOfPos*(numOfPos + 1)/2.0
        ySum = 0.0
        for i in range(len(scoreArray)):
            if classLabel[scoreArray[i]] == 1:
                delX = 0; delY = yStep;sumOfPos += (len(scoreArray) - i)
            else:
                delX = xStep; delY = 0; ySum += startPoint[1]
            
            plt.plot((startPoint[0],startPoint[0] + delX),(startPoint[1],startPoint[1]+delY), c='b')
          
            startPoint = [startPoint[0] + delX,startPoint[1]+delY]
        
        plt.plot((0.0,1.1),(0.0,1.1))
        
        print ("the Area Under the Curve is: ",(sumOfPos - sumOfNag)/(numOfNag*numOfPos))
        print ("the Area Under the Curve is: ",ySum*xStep)
        
        plt.show()
        '''
        
        '''   
        # second method to get num of pos and nag
        print(np.nonzero(classLabel == 1.0))
        print(len(np.nonzero(classLabel == 1.0)[0]))
        print(np.nonzero(classLabel == 0.0))
        print(len(np.nonzero(classLabel == -1.0)[0]))
        '''
        '''
        maxNum = [1,10,50,100,500]
        for i in maxNum:
            print("num of week classify : " ,i)
            model = DataUtil.loadModel("../../../data/boosting/model-%d"%(i))
            
            print(" the training Error of this model : ", model['trainError'])
            predict  = self.predictBoosting(dataIn, classLabel, model['tree'])
            
            errMat =  np.nonzero(predict + classLabel)[0]
            addError = 1.0 - len(errMat)/self.lines
            
            print(" the test Error of this model : ", addError)
        '''
        
        
        fig = plt.figure(1)
        fig.clf()
        maxNum = [1,10,50,100,500]
        color = ['b','red','black','g','yellow']
        for i in range(len(maxNum)):
            print("num of week classify : " ,i)
            model = DataUtil.loadModel("../../../data/boosting/model-%d"%(maxNum[i]))
            print(" the training Error of this model : ", model['trainError'])
            scores  = self.predictBoosting(dataIn, classLabel, model['tree'])
            predict  = np.sign(scores)
            errMat =  np.nonzero(predict + classLabel)[0]
            addError = 1.0 - len(errMat)/self.lines
            print(" the test Error of this model : ", addError)
            
            DataUtil.RocAuc(scores, classLabel, color[i])
        plt.plot((0.0,1.1),(0.0,1.1))
        plt.show()
       

ab = AdaBoosting()
#ab.trainModel()
ab.predict()