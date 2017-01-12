'''
Created on 2016年7月1日

@author: Ma
'''
import numpy as np;
import matplotlib.pyplot as plt;
import operator



class KNN :
    
    labelTrans = dict([("largeDoses",3),("smallDoses",2),("didntLike",1),
                  (3,"largeDoses"),(2,"smallDoses"),(1,"didntLike")]);
    
    vectorMatt = None;
    labelMatt = None;  
       
    def printTest(self):
        print("this is test!");
        
    def setMatt(self,vector,label):
        self.vectorMatt = vector;
        self.labelMatt = label;
    
    def loadDate(self,filePath):
        fr = open(filePath,'r');
        lines = fr.readlines();
        numOfLines = len(lines);
        vectorMatt = np.zeros((numOfLines,3));
        labelMatt = [];
        index = 0;
       
        for line in lines:
            line = line.strip();
            lineArray = line.split('\t');
            vectorMatt[index,:] = lineArray[0:3];
            label = lineArray[3];
            if  label in self.labelTrans :
                labelMatt.append(self.labelTrans.get(label) ) ;
            index += 1;
       
        return vectorMatt,labelMatt;
    
    def showPicture(self,x,y):
        fig = plt.figure();
        ax = fig.add_subplot(111);
        ax.scatter(x,y, 
                   50.0*np.array(self.labelMatt), 2.0*np.array(self.labelMatt));
        
        plt.xlabel('x');
        plt.ylabel('y');
        plt.title("by mpj");
        plt.show();
        
    def valueDivision(self,a,b):
        
        ctA = a.shape;
        ctB = b.shape;
        returnMatt = np.zeros(ctA);
        if ctA[0] == ctB[0] & ctA[1] == ctB[1]:
            print("true");
            for i in range(ctA[0]):
                for j in range(ctA[1]):
                    if b[i][j].any() != 0:
                        returnMatt[i][j]= a[i][j]/b[i][j];
                    else:
                        returnMatt[i][j]  = 0;
                    
        else:
            return "a,b not equal";
            
        return returnMatt;
        
    def norm(self,vector):
        minValue = vector.min(0);
        maxValue = vector.max(0);
        diffValue =maxValue  - minValue;
        colsNum = vector.shape[0];
        a = np.tile(diffValue,(colsNum,1));
        b = (vector - np.tile(minValue, (colsNum,1)));
        print(np.shape(a));
        print(np.shape(b));
        normValue = b/a
        
        return normValue;
    
    def classify(self,inputX,vectorMatt,labelMatt,k):
        #inputx : input vector
        #vectorMatt : vector of all data
        #labelMatt: list of all data, labelMatt vectorMatt one by one
        #k: k near
        cols = vectorMatt.shape[0];
        copyInputX = np.tile(inputX, (cols,1));
       
        distance = ((copyInputX - vectorMatt)**2).sum(1)**0.5;
        indicesSort = np.argsort(distance);
        countLabel = {};
        for i in range(k):
            latentLabel = labelMatt[indicesSort[i]];
            countLabel[latentLabel] = countLabel.get(latentLabel,0) + 1;
        labelSort = sorted(countLabel.items(),key = lambda x:(x[1]),  reverse = True);
        print(labelSort);
        return labelSort[0][0];
        
    def classifyTest(self,ratio,k):
        #ratio = 0.2 # ratio of training test
        #k = 600;
        # normalization data
        normMatt = self.norm(self.vectorMatt); 
        
        numOfDateset = normMatt.shape[0];
        numOfTestDataSet = int(numOfDateset*ratio);
        errorCount = 0.0;
        for i in range(numOfTestDataSet):
            classifyReturn = self.classify(normMatt[i,:], normMatt[numOfTestDataSet:,:], self.labelMatt[numOfTestDataSet:], k);
         
            
            print("the classifier came back with %s, the real anser is %s" 
                  % (self.labelTrans.get(classifyReturn),self.labelTrans.get(self.labelMatt[i])));
            if classifyReturn != self.labelMatt[i]:
                errorCount += 1.0;
            
        print("the total missclassifier error rate is : ", errorCount/numOfTestDataSet);
         
        
        
    
knn = KNN();
v,l = knn.loadDate("../../../data/KNN/datingTestSet.txt");
knn.setMatt(v,l);
knn.classifyTest(0.2,3);


        