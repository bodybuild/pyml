'''
Created on 2016年9月19日

@author: Ma
'''
import operator
import datetime
class treeNode:
    def __init__(self,nodeName,count,parentNode=None):
        self.name = nodeName
        self.count = count
        self.parentNode = parentNode
        self.nodeLink = None
        self.children = {}
    def Inc(self,count):
        self.count += count
    def Disp(self,ind=1):
        print(" "*ind,self.name," ",self.count)
        for child in self.children.values():
            child.Disp(ind + 1)

class FPgrowth:
    
    
    def loadData(self):
        simpDat = [['a', 'b'],
                   ['b', 'c', 'd'],
                   ['a', 'c', 'd', 'e'],
                   ['a', 'd', 'e'],
                   ['a', 'b', 'c'],
                   ['a', 'b', 'c', 'd'],
                   ['a'],
                   ['a', 'b', 'c'],
                   ['a', 'b', 'd'],
                   ['b', 'c', 'e']
                   ]
        return simpDat
    
    def initDataSet(self,simpData):
        initSet = {}
        for trans in simpData:
            initSet[frozenset(trans)] = initSet.get(frozenset(trans),0) + 1
        return initSet
    def updateNodeLink(self,rootNode,currentNode):
        while rootNode.nodeLink != None:
            rootNode = rootNode.nodeLink
        rootNode.nodeLink = currentNode
    def updateTree(self,orderItem,rootNode,itemTable,transSup):
        #print("updateTree ---- orderItem = " , orderItem)
        if orderItem[0][0] in rootNode.children:
            rootNode.children[orderItem[0][0]].Inc(transSup)
        else:
            rootNode.children[orderItem[0][0]] = treeNode(orderItem[0][0],transSup,rootNode)
            if itemTable[orderItem[0][0]][1] == None:
                itemTable[orderItem[0][0]][1] = rootNode.children[orderItem[0][0]]
            else:
                self.updateNodeLink(itemTable[orderItem[0][0]][1], rootNode.children[orderItem[0][0]])
        if len(orderItem) > 1:
            self.updateTree(orderItem[1::], rootNode.children[orderItem[0][0]], itemTable, transSup)
           
        

    def createFPTree(self,dataSet,minSup):
        itemTable1 = {}
        itemTable = {}
        # init k-1 item support
        for trans in dataSet:
            for scans in trans:
                itemTable1[scans] = itemTable1.get(scans,0) + dataSet[trans]
        
        # delete support less minSup
        for item in itemTable1.keys():
            if itemTable1[item] >= minSup:
                # itemTalbe sub-0 store k-1 itemSet and support, sub-1 store rootNode of link
                itemTable[item] = [itemTable1[item],None]
        
        # store frequent k-1 itemSet 
        k1Set = set(itemTable.keys())
        
        # init rootNode
        rootNode = treeNode("RootNode",1)
        
        for trans,transSup in dataSet.items():
            #sort item which both in  transaction && Frequent Set : Descent
            localItem = {}
            for scan in trans:
                if scan in k1Set:
                    localItem[scan] = itemTable[scan][0]
            #print("   itemTable = ", itemTable)
            if len(localItem) > 0:
                #print("  localItem = ", localItem)
                orderItem = sorted(localItem.items(), key=operator.itemgetter(1), reverse = True)
                
                #print("  orderItem = " ,orderItem)
                # update tree with a transaction
                self.updateTree(orderItem, rootNode, itemTable, transSup)
        return rootNode,itemTable
    
    def findPath(self,node):
        prefixPath = []
        while node.parentNode != None:
            prefixPath.append(node.name)
            node = node.parentNode
        return prefixPath[1::]
    def findPrefixPath(self,baseItem,itemTable):
        rootLink = itemTable[baseItem][1]
        conPatterBase = {}
        while rootLink != None:
            #print("name = ", rootLink.name ," count = ",rootLink.count)
            prefixPath = self.findPath(rootLink)
            #print("prefixPath = ",prefixPath)
            if len(prefixPath) >= 1:
                conPatterBase[frozenset(prefixPath)] = rootLink.count
            rootLink = rootLink.nodeLink
    
        return conPatterBase
    def miningFI(self,itemTable,itemBase, freqItemSet , transSup):
        orderItem = {}
        for item in itemTable.keys():
            orderItem[item] = itemTable[item][0]
        #order k-1 item , Decent
        orderItem = [ v[0] for v in sorted(orderItem.items(), key=lambda p:p[1])]
        print(" **************  oderItem = " , orderItem)
        for item in orderItem:
            freqitemBase = itemBase.copy()
            freqitemBase.add(item)
            freqItemSet.append(freqitemBase)
            print("  item = ",item, " itemBase = " ,freqitemBase)
            print("  freqItemSet = ", freqItemSet)
            prefixPath = self.findPrefixPath(item, itemTable)
            print("  prefixPath = " ,prefixPath)
            if len(prefixPath) >= 1:
                rootNode,headTable = self.createFPTree(prefixPath, transSup)
                rootNode.Disp()
                if headTable != None:
                    self.miningFI(headTable, freqitemBase, freqItemSet, transSup)
    def _run(self):
        simpData = self.loadData()
        dataSet = self.initDataSet(simpData)
        rootNode,itemTable = self.createFPTree(dataSet, 3)
        rootNode.Disp()
        transSup = 2;itemBase = set([]);freqItemSet = []
        self.miningFI( itemTable,itemBase, freqItemSet , transSup)
        print(freqItemSet)

fp = FPgrowth()
#fp._run()

currentTime = datetime.datetime.now().microsecond
dataPath = "../../../data/FP-growth/kosarak.dat"
fw = open(dataPath,'r')
lines = fw.readlines()
primData = [ line.split() for line in lines]

minSup = 100000
dataSet = fp.initDataSet(primData)
rootNode,itemTable = fp.createFPTree(dataSet, minSup)

rootNode.Disp()

print("the time cost for built FP-tree is : ", (datetime.datetime.now().microsecond - currentTime)/1000)

itemBase = set([]);freqItemSet = []
fp.miningFI( itemTable,itemBase, freqItemSet , minSup)
print(freqItemSet)
print("the time cost for built FP-tree and mining Frequent Item Set is  : ", (datetime.datetime.now().microsecond - currentTime)/1000)




