'''
Created on 2016年7月20日

@author: Ma
'''
from machineLearning.logistic import logistic;
import numpy as np;
import matplotlib.pyplot as plt;

class horsePredic:
    
    
    
    
    def classifyVector(self,inputX,weight):
        prob = logistic.Logistic.sigmod(inputX*weight);
        if prob > 0.5:
            return 1.0;
        else:
            return 0.0;
    def horsePredictStochastic(self,path,alpha,numIter=300):
        lg = logistic.Logistic();
        weigth,x,y = lg.stochasticGradientDescent(path, 0  ,alpha, numIter);
        return weigth;
    
    def horsePredictBatch(self,path,alpha,numIter=300):
        lg = logistic.Logistic();
        weigth,x,y = lg.batchGradientDescent(path, 0  , alpha, numIter);
        return weigth;
    def computeRoll(self,true_p, true_f, classVec):
        total = len(classVec);
        positive = sum(classVec);
        precise = float(true_p/(total - positive - true_f + true_p));
        recall = float(true_p/positive);
        return precise,recall;
    
    def predict(self,method,alpha=0.001,numIter =500):
        pathTraining = "../../../data/logistic/horseColicTraining.txt";
        pathTest = "../../../data/logistic/horseColicTest.txt";
        dataVec,labelVec =  logistic.Logistic.loadDate(pathTest);
        lines = len(dataVec);
        x1 = np.ones((lines,1)); #增加 截距
        dataMat = np.mat(dataVec, dtype = np.float);
        dataMat = np.column_stack((x1,dataMat));
        
        if method == "batch":
            weigth = self.horsePredictBatch(pathTraining,alpha,numIter);
        else: 
            weigth = self.horsePredictStochastic(pathTraining,alpha,numIter);
        
        true_f = 0.0;
        true_p = 0.0;
        for i  in range(len(dataMat)):
            classify = self.classifyVector(dataMat[i], weigth);
            
            #print(" the result come bask with %f, the real class is %f" %(classify,labelVec[i]));
            if classify == labelVec[i]:
                if classify == 1 or classify == "1":
                    true_p += 1;
                else:
                    true_f += 1;
     
        precise,recall =  self.computeRoll(true_p, true_f, labelVec);
        #print("the precision is %f, the recall is %f "% (precise,recall));
        return precise,recall;
      
    def _run(self):
        """
        self.predict("batch",alpha=0.01,numIter=400);
        print("************************")
        self.predict("batch",alpha=0.01,numIter=1000);
        print("************************")
        self.predict("batch",alpha=0.01,numIter=5000);
        #self.predict("stochastic");
        """
        x = [];
        y = [];
        z = [];
        for i in range(10):
            alpha = 0.1/(i*5+1);
            precise,recall = self.predict("batch", alpha, 1500);
            z.append(i);
            x.append(precise);
            y.append(recall);
            print(precise);
            print(recall);
        
        plt.figure(1);
        plt.title("iterator");
        plt.plot(z,x);
        plt.plot(z,y);
        plt.show();
        
            

hp = horsePredic();
hp._run();
        