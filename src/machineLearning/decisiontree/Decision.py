'''
Created on 2016年7月7日

@author: Ma
'''
import numpy as np;
import operator;
import pickle;
class Decision:
    def loadDate(self,datePath):
        fr = open(datePath,"r");
        vectorMatt = [];
        for inn in fr.readlines():
            if inn.strip():
                vectorMatt.append(inn.strip().split("\t")); 
        fr.close();
        return vectorMatt;
    def splitData(self,data,label,axis,value):
        reDateMatt = [];
        splitVec = [];
        for line in data:
            if line[axis] == value:
                splitVec = line[0:axis];
                splitVec.extend(line[axis + 1:]);
                reDateMatt.append(splitVec);
        splitVec = label[0:axis];
        splitVec.extend(label[axis + 1:]);
        reAttrMatt = splitVec;
        
        return reDateMatt,reAttrMatt;
        
    def calEntropy(self,vectorMatt):
        numOfData = len(vectorMatt);
        if numOfData == 0:
            return 0; 
        labelMap = {};
        entropyCal = 0.0;
        for line in vectorMatt:
            if line[-1] not in labelMap.keys():
                labelMap[line[-1]] = 0;
            labelMap[line[-1]] += 1;
        for key in labelMap.keys():
            prob = float(labelMap[key]/numOfData);
            entropyCal += prob*(np.log(prob));
        return -1*entropyCal;
    
    def calFeatureEntropy(self,dateset,attrset,axis):
        featureValue = [col[axis] for col in dateset];
        uniqueFeatureValue = set(featureValue);
        returnEntropy = 0.0;
        for feature in uniqueFeatureValue:
            featureData,attr = self.splitData(dateset,attrset, axis, feature);
            prob = float(len(featureData)/len(dateset));
            featureEntropy = self.calEntropy(featureData);
            returnEntropy += prob*featureEntropy;
        return returnEntropy;
                
    def chooseBestSplitFeature(self,dateset,attrset):
        bestInfo = 0.0;
        bestFeature = -1;
        entropBase = self.calEntropy(dateset);
        numOfFeature = len(dateset[0]) - 1;
        for i in range(numOfFeature):
            subEntropy = self.calFeatureEntropy(dateset,attrset, i);
            infoGain = entropBase - subEntropy;
            if infoGain > bestInfo:
                bestFeature = i;
                bestInfo = infoGain;
        return bestFeature; 
    
    def majorityCnt(self,dateset):
     
        colMap = {};
        for date in dateset:
            if date not in colMap.keys():
                colMap[date] = 0;
            colMap[date] +=1;
        return sorted(colMap.items(), key=operator.itemgetter(1) , reverse = True)[0][0];
            
    
    def createTree(self,vectorMatt,attrMatt):
        labelList = [line[-1] for line in vectorMatt];
        if labelList.count(labelList[0]) == len(labelList): #if all date has been split into one class          
            return labelList[0];
        if len(vectorMatt[0]) == 1:  # if all attribute has split
            return self.majorityCnt(vectorMatt);
        
        bestFeatureIndex = self.chooseBestSplitFeature(vectorMatt,attrMatt);
        bestFeature = attrMatt[bestFeatureIndex];
        
        myTree = {bestFeature:{}};
        
        featureValue = set([line[bestFeatureIndex] for line in vectorMatt]);
    
       
        for feature in featureValue:
            v,l = self.splitData(vectorMatt, attrMatt,bestFeatureIndex, feature);
            myTree[bestFeature][feature] = \
                self.createTree(v,l);
       
        return myTree;
    
    def testClassify(self,myTree,testVec,attrLabel):
      
        firstAttr  = [line for line in  myTree.keys()][0];
      
        attrIndex = attrLabel.index(firstAttr);
       
        testValue = testVec[attrIndex];
        secondDict = myTree.get(firstAttr)[testValue];
    
        if isinstance(secondDict, dict):
            return self.testClassify(secondDict, testVec, attrLabel);
        else:
            return secondDict;
            
    def classify(self,myTree, testVec, attrLabel):
        mis_classify = 0.0;
        for i in range(len(testVec)):  
            classLabel = self.testClassify(myTree, testVec[i], attrLabel);
            print("The class of line %d come back with %s, The real class is %s. " %(i,classLabel,testVec[i][-1]));
            if classLabel != testVec[i][-1]:
                mis_classify += 1.0;
        print("The total mis classify rate is ", mis_classify/len(testVec));
            
    
    
    def prunning(self,myTree):
        print("REP: reduced error pruning");
        print("CCP: cost-complexity pruning");
        print("MEP: minimum error pruning");
        
        
    
    def storeTree(self,myTree,fileName):
        fw = open(fileName,"wb");
        pickle.dump(myTree,fw);
        fw.close();
    
    def loadTree(self,fileName):
        fr = open(fileName,"rb");
        return pickle.load(fr);
        
    def _run(self):
        vectorMatt = self.loadDate("../../../data/decisionTree/lenses.txt");
        print(vectorMatt);
        attrMatt = self.loadDate("../../../data/decisionTree/attribution.txt")[0];
        print(attrMatt);
        tree = self.createTree(vectorMatt, attrMatt);
        print(tree);
        self.storeTree(tree, "../../../data/decisionTree/tree");
        print(self.loadTree("../../../data/decisionTree/tree"));
        self.classify(tree,vectorMatt,attrMatt);
       
dd = Decision();
dd._run();