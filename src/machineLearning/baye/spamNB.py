'''
Created on 2016年7月19日

@author: Ma
'''
from nt import listdir;
import re;
import numpy as np;
from machineLearning.baye import Bayes
class spamNB:
    
    def loadFile(self,path,flag):
        docList = [];
        classVec = [];
        fileList = listdir(path);
        
        for file in fileList:
            fw = open(path + "/" + file,"rb");
            doc = fw.read();
            docMatt = self.parseText(doc);
            docList.append(docMatt);
            classVec.append(flag);
        return docList,classVec;

    def parseText(self,line):
        lineArray = re.split(r'\W+', line.decode());
        return [line.lower() for line in lineArray if len(line) >= 3 ];
    
    def docToVec(self,doc,vocabulary):
        if len(doc) == 0 : return "error!";
        docVecMatt = np.zeros(len(vocabulary));
        for word in doc:
            if word in vocabulary:
                docVecMatt[vocabulary.index(word)] += 1;
        return docVecMatt;
    
    
    
    def spamModel(self):
        bayes = Bayes.NavieBayes();
        hamList,hamClass = self.loadFile("../../../data/bayes/email/ham",0);
        spamList,spamClass = self.loadFile("../../../data/bayes/email/spam",1);
     
        classVec = [];
        docList = np.concatenate((hamList,spamList));
        classVec.extend(hamClass);
        classVec.extend(spamClass);
        vocabulary = bayes.createVocabulary(docList);
        docMatt = [];
        for doc in docList:
            docMatt.append(self.docToVec(doc, vocabulary));
        print(docMatt);
        
        NavieBayesModel = bayes.trainNB(docMatt, classVec,1); 
        bayes.storeModel(NavieBayesModel, "../../../data/bayes/email/bayes");
        bayes.storeModel(vocabulary,"../../../data/bayes/email/vocabulary");
    
    def loadTestFile(self,path):
        docList = [];
        classVec = [];
        fileList = listdir(path);
        
        for file in fileList:
            fw = open(path + "/" + file,"rb");
            doc = fw.read();
            docMatt = self.parseText(doc);
            docList.append(docMatt);
            classVec.append(int(re.split("[,_.]", file)[1]));
        return docList,classVec,fileList;
    
    def computeRoll(self,true_p, true_f, classVec):
        total = len(classVec);
        positive = sum(classVec);
        precise = float(true_p/(total - positive - true_f + true_p));
        recall = float(true_p/positive);
        return precise,recall;
    def testSpamNB(self):
        bayes = Bayes.NavieBayes();
        testList,classVec,fileList = self.loadTestFile("../../../data/bayes/email/test");
        model = bayes.loadModel("../../../data/bayes/email/bayes");
        vocabulary = bayes.loadModel("../../../data/bayes/email/vocabulary");
        true_f = 0.0;
        true_p = 0.0;
        for i in range(len(testList)):
            testVec = self.docToVec(testList[i], vocabulary);
            classify = bayes.classifyNB(model, testVec);
           
            print(" the email named %s, come back with %s, And real classify is %s" %
                  (fileList[i], "spam"  if classify == 1 or classify == "1" else "ham",
                   "spam"  if classVec[i] == 1 or classVec[i] == "1" else "ham"));
            
            if classify == classVec[i]:
                if classify == 1 or classify == "1":
                    true_p += 1;
                else:
                    true_f += 1;
                
           
        precise,recall = self.computeRoll(true_p, true_f, classVec);
        print("the precision is %f, the recall is %f "% (precise,recall));
    def _run(self):
        #self.spamModel();
        self.testSpamNB();
   
sp = spamNB();
sp._run();