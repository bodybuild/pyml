'''
Created on 2016年7月10日

@author: Ma
'''
import matplotlib.pyplot as plt;
import pickle;
class DrawTree:
    
    leafNode = dict(boxstyle="round4,pad=.5",fc = "0.5");
    decisionNode = dict(boxstyle="round4",fc = "0.8"); 
    arrowType = dict(arrowstyle = "<-");
    
    def plotNode(self,nodeText,nodePoint,parentPoint,nodeType):
        plt.annotate(nodeText ,xytext = nodePoint,xy = parentPoint,arrowprops = self.arrowType,bbox = nodeType,);
    
    def plotMidText(self,cntrPt, parentPt, txtString):
        xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0];
        yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1];
     
        plt.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
   
    def getTreeDeep(self,Tree):
        childDict = Tree.get(list(Tree.keys())[0]);
        maxDeep = 0;
       
        for child in childDict.keys():
            deep = 1;
            if isinstance(childDict[child], dict):
                deep += self.getTreeDeep(childDict[child]);
            else:
                deep += 1;
            
            if maxDeep < deep:maxDeep = deep;
        return maxDeep;
    def getNumOfLeafs(self,Tree):
        numOfLeafs = 0;
        childDict = Tree.get(list(Tree.keys())[0]);
        for child in childDict.keys():
            if isinstance(childDict[child], dict):
                numOfLeafs += self.getNumOfLeafs(childDict[child]);
            else:
                numOfLeafs += 1;
        return numOfLeafs;
   
    def loadTree(self,fileName):
        fr = open(fileName,"rb");
        return pickle.load(fr);
    
    def plotTree(self,Tree,deep,width,offset):
     
        rootKey = list(Tree.keys())[0];
        childDict = Tree.get(rootKey);
        rootPoint = (float(width/2) + offset,deep);
        plt.scatter(float(width/2) + offset,deep);
        for childKey in childDict:
            child = childDict[childKey];
           
            if isinstance(child, dict):
                maxWidth = self.getNumOfLeafs(child);
                self.plotMidText((float(maxWidth/2) + offset,deep - 1), rootPoint, childKey);
                self.plotNode(list(child.keys())[0],(float(maxWidth/2) + offset,deep - 1), rootPoint, self.decisionNode);
                self.plotTree(child, deep - 1, maxWidth,offset);
                offset += maxWidth*2;
            else:
                self.plotMidText((offset,deep - 1), rootPoint, childKey);
                self.plotNode(child, (offset,deep - 1), rootPoint, self.leafNode);
                plt.scatter(offset,deep - 1);
                offset += 1*2;
                
       
        
    def creatTree(self,Tree):
        deep = self.getTreeDeep(Tree);
        width = self.getNumOfLeafs(Tree);
        self.plotNode(list(Tree.keys())[0],(float(width/2),deep), (float(width/2),deep), self.decisionNode);
                
        self.plotTree(Tree,deep,width,0);
        plt.show();
    def _run(self):
        Tree = self.loadTree("../../../data/decisionTree/tree");
        self.creatTree(Tree);
        """
        plt.figure(1);
        plt.clf();
        self.plotNode("test",(0,0),(0,0),self.leafNode);
        plt.show();
        """


dd = DrawTree();
dd._run();    