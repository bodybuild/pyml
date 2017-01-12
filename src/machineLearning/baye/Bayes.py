'''
Created on 2016年7月18日

@author: Ma
'''

import numpy as np;
from builtins import set
import pickle;
from array import array
import operator;


class NavieBayes:
    
    def loadDate(self):
        docList = [['my', 'dog','dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']];
        classVec = [1,2,3,2,4,3];    #1 is abusive, 0 not
        return docList,classVec;
    
    def createVocabulary(self,docList):
        vocabulary = set([]);
   
        for doc in docList:
            vocabulary = vocabulary | set(doc);
        return list(vocabulary);
    
    def docToVec(self,doc,vocabulary):
        if len(doc) == 0 : return "error!";
        docVecMatt = np.zeros(len(vocabulary));
        for word in doc:
            if word in vocabulary:
                docVecMatt[vocabulary.index(word)] = 1;
            else:
                docVecMatt[vocabulary.index(word)] = 0;
        return docVecMatt;
                    
    def trainNB(self,docVecMatt,classVec,laplace):
                
        numOfTrainSet = len(classVec);
        numOfDoc = len(docVecMatt[0]);
        classDict = {};
        
        # compute probobility of  class , vector of class,
        for i in range(len(classVec)):
            if classVec[i] in classDict.keys():
                classDict[classVec[i]]['prob'] += 1;
            else:
                classDict[classVec[i]] = dict(prob= 1,vec= np.zeros(numOfDoc),total=0);
           
            classDict[classVec[i]]['vec'] += docVecMatt[i];
            classDict[classVec[i]]['total'] += 1;
            
            print(classDict[classVec[i]]['total'])  
        print(numOfDoc)
        numOfclass = len(classDict.keys()); 
        laplaceVec = [1*laplace]*numOfDoc;
        print(classDict);
        for child in classDict.keys():
            classDict[child]['prob'] = float( ( classDict[child]['prob'] + laplace*1)/( numOfTrainSet + laplace*numOfclass));
            classDict[child]['vec'] = (classDict[child]['vec'] + laplaceVec)/(classDict[child]['total'] + 2);
        print(classDict)
        return classDict;
    
    def storeModel(self,classDict,path):
        fw = open(path,"wb");
        pickle.dump(classDict,fw);
        fw.close();
    
    def loadModel(self,path):    
        fr = open(path,"rb");
        model = pickle.load(fr);
        return model;
    
    def classifyNB(self,model,dateSet):
        classifyDict = {};
       
        for classify in model.keys():
            probValue = 1.0;
            for i in range(len( dateSet)):
                if dateSet[i] == 1:
                    probValue =  probValue*model[classify]['vec'][i]; 
            probValue *= model[classify]['prob'];
            classifyDict[classify] = probValue;
        print(classifyDict);
        return sorted(classifyDict.items(), key=operator.itemgetter(1), reverse = True)[0][0];
        
    def testNB(self):
        testEntry = ['love', 'my', 'dog'];
        model = self.loadModel("../../../data/bayes/bayes");
        vocabulary = self.loadModel("../../../data/bayes/vocabulary");
        testVec = self.docToVec(testEntry, vocabulary);
        classify = self.classifyNB(model, testVec);
        print("the classifier come back with  ",classify);
    
    def _run(self):
        
        docList,classVec = self.loadDate();
        vocabulary = self.createVocabulary(docList);
        print(vocabulary);
        docVecMatt = [];
        for doc in docList:
            print(doc);
            docVecMatt.append(self.docToVec(doc, vocabulary));
        print(docVecMatt);
    
        NavieBayesModel = self.trainNB(docVecMatt, classVec,1); 
        self.storeModel(NavieBayesModel, "../../../data/bayes/bayes");
        self.storeModel(vocabulary,"../../../data/bayes/vocabulary");
       
        self.testNB();
        
        
nb = NavieBayes();
nb._run(); 