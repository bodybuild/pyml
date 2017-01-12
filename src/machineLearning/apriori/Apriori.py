'''
Created on 2016年9月6日

@author: Ma
'''

import numpy as np
import matplotlib.pyplot as plt
import time
class Apriori:
    
    def loadData(self):
        return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
    
    def initSet(self,dataSet):
        initSet = []
        for item in dataSet:
            for ele in item:
                if frozenset([ele]) not in initSet:
                    initSet.append(frozenset([ele]))
        initSet.sort()
        return initSet

    def calSupport(self,dataSet,freSet,minSupport = 0.6):
        freSupport = {}
        for item in dataSet:
            for can in freSet:
                if not bool(set(can) - set(item)):
                    if can not in freSupport.keys():
                        freSupport[can] = 1
                    else:
                        freSupport[can] += 1
        dataLen = len(dataSet)
        returnList = []
        supportList = {}
        for key in freSupport:
            fre = float(freSupport[key])
            support = fre/dataLen

            if support >= minSupport:
                returnList.append(key)
                supportList[key] = support
        return returnList,supportList
    
    def aprioriMerge(self,freSet,k):
        setLen = len(freSet)
        tmpList = []
        for i in range(setLen):
            for j in range(i + 1,setLen):
                L1 = list(freSet[i])[:k-2]
                L2 = list(freSet[j])[:k-2]
                L1.sort();L2.sort()
              
                if L1 == L2 :
                    tmpList.append(freSet[i]|freSet[j])
        return tmpList
    
    def apriori(self,dataSet,sup):
        C1 =  self.initSet(dataSet)
        Lk,supportLK = self.calSupport(dataSet, C1, sup)
        k = 2
        LastFre = [Lk]
        while True:
            Ck = self.aprioriMerge(LastFre[k - 2], k)  
            Lk,supLK = self.calSupport(dataSet, Ck, sup) 
            if len(Lk) > 0: 
                LastFre.append(Lk)
                supportLK.update(supLK)
            else:
                break
            k += 1
        return LastFre,supportLK
    
    def ruleLoop(self,fre,oneEle,supportLk,minConf):
       
        returnList = []
        while len(fre) >= len(oneEle[0]) + 1:
            print("**********len of element in oneEle is  ",len(oneEle[0]))
            returnList.extend(self.calConf(fre, [ fre - ele for ele in oneEle], supportLk, minConf))
            oneEle = self.aprioriMerge(oneEle, 2)
         
        return returnList
    def calConf(self,fre,oneEle,supportLk,minConf):
        returnList = []
        for ele in oneEle:
           
            conf = supportLk[fre]/supportLk[fre - ele]
            if conf > minConf:
                print("rule : " , fre - ele, " ---------> ", ele , " , conf = ",conf)
                returnList.append((fre-ele,ele,conf))
        return returnList
    def generateRules(self,Lk,supportLk,minConf=0.7):
        returnRule = []
        
        for i in range(1,len(Lk)):
            print("i = " ,i)
            for fre in Lk[i]:
                oneEle = [ frozenset([ele]) for ele in list(fre)]
                if i > 1:
                    returnRule.extend(self.ruleLoop(fre,oneEle,supportLk,minConf))
                else:
                    returnRule.extend(self.calConf(fre,oneEle,supportLk,minConf))
        return returnRule
    def _run(self):
        #filePath = "../../../data/Apriori/bills20DataSet.txt"
        '''
        dataSet = self.loadData()
        Lk,supportLK = self.apriori(dataSet)
        print(Lk)
        print(supportLK)
        ruleList =   self.generateRules(Lk, supportLK, 0.6)
        print(ruleList)
        #self.calSupport(dataSet, initSet)
        '''
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
        sup = float(2/len(simpDat))
        Lk,supportLK = self.apriori(simpDat,sup)
        print(Lk)
        print(supportLK)
        print(len(Lk[0]))
        print(len(Lk[1]))
        print(len(Lk[2]))
      
        
apriori = Apriori()
apriori._run()