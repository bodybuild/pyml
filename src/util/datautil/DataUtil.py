'''
Created on 2016年7月22日

@author: Ma
'''

import numpy as np
import matplotlib.pyplot as plt
from nt import listdir
from matplotlib.patches import Circle
class DataUtil:
    
    @staticmethod
    def loadDateMatt(path,dataType = float):
        
        fileMatt = np.loadtxt(path,dtype = dataType);
      
        return fileMatt;
    
    @staticmethod
    def loadDateFloat(path,dataType = float):
        
        fileMatt = np.loadtxt(path,dtype = dataType);
      
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
    def plotScatter(path):
        dataMaker = {-1:"x",0:"^",1:"o",2:"1",3:"+",4:"h",5:"8",6:"s"};
        dataVec,labelVec = DataUtil.loadDateFloat(path, np.float);
        
        for i in range(len(labelVec)):

            plt.scatter(dataVec[i,0],dataVec[i,1],
                        s=50,c = (labelVec[i] + 2)*50, marker=dataMaker.get(labelVec[i]));
    
    @staticmethod
    def getXrange(dataIn):
        mid = (min(dataIn) + max(dataIn))/2;
        Xrange = max(dataIn) - min(dataIn);
        return mid - Xrange/5,mid + Xrange/5;
    @staticmethod
    def pltPlot(weigth,text,x1 = -5,x2=5):
        print(text);
        x = np.linspace(x1,x2,20);
        plt.plot(x,-1*(weigth[0] + weigth[1]*x)/weigth[2]);
        plt.annotate(text,xy=(x1,-1*(weigth[0] + weigth[1]*(x1))/weigth[2]),xytext = (-40,30),
            textcoords='offset points',arrowprops=dict(arrowstyle="->",connectionstyle="arc,angleA=0,armA=30,rad=10"));
        plt.xlabel("x1");
        plt.ylabel("x2");
    
    @staticmethod
    def pltCycle(xy,radius):
        ax = plt.gca();
        
        circle = Circle(xy,radius,facecolor='none', edgecolor=(0,0.0,0.0), linewidth=3,alpha = 0.5);
        ax.add_patch(circle);
        return circle;
    @staticmethod
    def storeModel(classDict,path):
        import pickle;
        fw = open(path,"wb");
        pickle.dump(classDict,fw);
        fw.close();
    @staticmethod
    def loadModel(path):    
        import pickle;
        fr = open(path,"rb");
        model = pickle.load(fr);
        return model;  
    
    @staticmethod
    def plotShow():
        plt.show();
        
    @staticmethod    
    def imgToVector(fileName):
        imgMatt = np.zeros((1,1024));
        fr = open(fileName,'r');
        lines = fr.readlines();
        numOfLines = len(lines);
        
        for i in range(numOfLines):
            for j in range(numOfLines):
                imgMatt[0,i*32 + j] = int(lines[i][j]);
        return imgMatt;
    
    @staticmethod
    def loadFile(filePath):
        fileList = listdir(filePath);
        numOfTraning = len(fileList);
        trainVectorMatt = np.zeros((numOfTraning,1024));
        trainLabelMatt = np.zeros(numOfTraning);
        for i in range(numOfTraning):
            labelIndex = fileList[i].find("_");
            trainLabelMatt[i] = fileList[i][0:labelIndex];
            trainVectorMatt[i,] = DataUtil.imgToVector(filePath + "/" +  fileList[i]);
        
        return trainVectorMatt,trainLabelMatt;
    
    @staticmethod
    def RocAuc(scores,classLabel,color = 'b'):
        scoreArray = np.argsort(-1*scores.T).reshape(-1)
        numOfPos = np.sum( classLabel == 1.0 )
        numOfNag = np.sum( classLabel == -1.0 )
        
        yStep = float(1.0/numOfPos)
        xStep = float(1.0/numOfNag)
        
        
     
        startPoint = [0.0,0.0]
        ySum = 0.0
        for j in range(len(scoreArray)):
            if classLabel[scoreArray[j]] == 1:
                delX = 0; delY = yStep;
            else:
                delX = xStep; delY = 0; ySum += startPoint[1]
            
            plt.plot((startPoint[0],startPoint[0] + delX),(startPoint[1],startPoint[1]+delY), c=color)
            startPoint = [startPoint[0] + delX,startPoint[1]+delY]
        print ("the Area Under the Curve is: ",ySum*xStep)
        
        