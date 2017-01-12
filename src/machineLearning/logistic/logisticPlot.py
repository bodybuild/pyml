'''
Created on 2016年7月19日

@author: Ma
'''
import matplotlib.pyplot as plt
from machineLearning.logistic.logistic import Logistic;
import numpy as np
from matplotlib.font_manager import weight_as_number
class LogisticPlot:
    
   
  
    def plotScatter(self):
        lg = Logistic();
        dataVec,labelVec = lg.loadDate("../../../data/logistic/testSet.txt");
        plt.scatter(dataVec[:,0],dataVec[:,1],s=50, c = labelVec*1 );
        
    
    def pltPlot(self,weigth,text):
        x = np.linspace(-5.0,5.0,20);
        plt.plot(x,-1*(weigth[0] + weigth[1]*x)/weigth[2]);
        plt.annotate(text,xy=(-4.0,-1*(weigth[0] + weigth[1]*(-4.0))/weigth[2]),xytext = (-40,30),
            textcoords='offset points',arrowprops=dict(arrowstyle="->",connectionstyle="arc,angleA=0,armA=30,rad=10"));
        plt.xlabel("x1");
        plt.ylabel("x2");
        
    @staticmethod
    def plotWeigth(x,y):
      
        yMat = np.mat(y);
        
        plt.ylabel("X0");
        plt.plot(x,list(np.asarray(yMat[:,0]).reshape(-1)));
        
        plt.ylabel("X1");
        plt.plot(x,list(np.asarray(yMat[:,1]).reshape(-1)));
        
        plt.ylabel("X2");
        plt.plot(x,list(np.asarray(yMat[:,2]).reshape(-1)));
        
    def picturePlot(self,weigth,weigthX,weigthY,weigthFlag):
        
        numOfPicture = len(weigth);
        print(numOfPicture);
        plt.figure(1);
        for i in range(numOfPicture):
            ax =  plt.subplot(numOfPicture,2,(i*2 + 1))
            plt.sca(ax);
           
            self.plotScatter();
            self.pltPlot(np.asarray(weigth[i]).reshape(-1),weigthFlag[i]);
            
            ax1 =  plt.subplot(numOfPicture,2,(i*2 + 2))
            plt.sca(ax1);
            self.plotWeigth(weigthX[i], weigthY[i]);
        
        plt.show();
    def createPlot(self):
        lg = Logistic();
        weigth = [];
        weigthX = [];
        weigthY = [];
        weigthFlag = [];
        threshold = 0;
        alpha = 0.01;
        numIter = 500;
       
        weigthsB,weigthsBX,weigthsBY = lg.batchGradientDescent( 0, 0.001,1000);
        weigthsS,weigthsSX,weigthsSY = lg.stochasticGradientDescent( 0, 0.1);
      
        weigthImproveFF,weigthImproveXFF,weigthImproveYFF = lg.improveSGD(threshold,alpha, numIter,randomFlag=False,alphaFlag=False);
        weigth.append(weigthImproveFF);
        weigthX.append(weigthImproveXFF);
        weigthY.append(weigthImproveYFF);
        weigthFlag.append("0.01F");
        
        weigthImproveFF,weigthImproveXFF,weigthImproveYFF = lg.improveSGD(threshold,0.1, numIter,randomFlag=False,alphaFlag=False);
        weigth.append(weigthImproveFF);
        weigthX.append(weigthImproveXFF);
        weigthY.append(weigthImproveYFF);
        weigthFlag.append("0.001F");
        
        """
        weigthImproveTF,weigthImproveXTF,weigthImproveYTF = lg.improveSGD(threshold,alpha, numIter,randomFlag=True,alphaFlag=False);
        weigth.append(weigthImproveTF);
        weigthX.append(weigthImproveXTF);
        weigthY.append(weigthImproveYTF);
        weigthFlag.append("randomT&&alphaF");
        """
        weigthImproveFT,weigthImproveXFT,weigthImproveYFT = lg.improveSGD(threshold,alpha, numIter,randomFlag=False,alphaFlag=True);
        weigth.append(weigthImproveFT);
        weigthX.append(weigthImproveXFT);
        weigthY.append(weigthImproveYFT);
        weigthFlag.append("0.01T");
        weigthImproveFT,weigthImproveXFT,weigthImproveYFT = lg.improveSGD(threshold,0.1, numIter,randomFlag=False,alphaFlag=True);
        weigth.append(weigthImproveFT);
        weigthX.append(weigthImproveXFT);
        weigthY.append(weigthImproveYFT);
        weigthFlag.append("0.001T");
        
        """
        weigthImproveTT,weigthImproveXTT,weigthImproveYTT = lg.improveSGD(threshold,alpha, numIter,randomFlag=True,alphaFlag=True);
        weigth.append(weigthImproveTT);
        weigthX.append(weigthImproveXTT);
        weigthY.append(weigthImproveYTT);
        weigthFlag.append("randomT&&alphaT");
        """
        self.picturePlot(weigth, weigthX, weigthY, weigthFlag);
      
    def _run(self):
        self.createPlot();
        
       
        
        

lp = LogisticPlot();
lp._run();