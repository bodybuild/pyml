'''
Created on 2016年8月17日

@author: Ma
'''
import numpy as np

from util.datautil.DataUtil import DataUtil 
from numpy.core import umath
class SMOMpj:
    def __init__(self,dataIn,classLabel,C = 0.5 ,toler= 0.001,maxIter = 50,threshold = 0.0001,method=("linear",0)):
  
        self.dataIn = np.mat(dataIn)
        self.classLabel = np.mat(classLabel).T
        self.C = C  # penalty factor 
        self.toler = toler #error tolerance
        self.maxIter = maxIter
        self.threshold = threshold # lagrange multipliers
        self.b  = 0 # hyperplane params
        self.lines,self.cols = np.shape(self.dataIn)
        self.alpha = np.zeros((self.lines,1))
        self.eCache = np.zeros((self.lines,2))
        self.method = method
        self.K = np.mat(np.zeros((self.lines,self.lines)))
        self.weigth = np.zeros(self.lines)
        for i in range(self.lines):
            self.K[:,i] = self.kernelTrans(self.dataIn, self.dataIn[i,:], self.method)
      
    def calWeigth(self):
        weigth = (np.multiply(self.alpha,self.classLabel).T)*self.dataIn; 
        return weigth
    @staticmethod
    def kernelTrans(dataIn,X,method):
        lines,cols = np.shape(dataIn)
        k = np.zeros((lines,1))
        if(method[0] == "linear"):
            k = dataIn*X.T
        elif(method[0] == "gaussian"):
            for i in range(lines):
                k[i] = (dataIn[i] - X)*(dataIn[i] - X).T
           
            k = umath.exp(k/((-1)*method[1]**2))
        else: 
            raise NameError('Houston We Have a Problem -- \
        That Kernel is not recognized')
       
        return k;
   
    def Gx(self,i):
        Gx = float((np.multiply(self.alpha,self.classLabel)).T*self.K[:,i] + self.b);# g(x)
        return Gx;
    
    def computeLoss(self,i):
        return self.Gx(i) - self.classLabel[i]
   
  
    def selectJrand(self,i,lines):
        j = i;
        while j == i:
            j = int(np.random.uniform(0,lines));
        return j;
    
    def computeLH(self,i,j):
        
        if self.classLabel[i] == self.classLabel[j]:
            L = max(0,self.alpha[i] + self.alpha[j] - self.C);
            H = min(self.C,self.alpha[i] + self.alpha[j]);
        else:
            L = max(0,self.alpha[j] - self.alpha[i]);
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i]);
        return L,H;
    
    def selectJ(self,i):
        Ei = self.computeLoss(i)
        if (self.classLabel[i]*Ei > self.toler and self.alpha[i] > 0) \
            or (self.classLabel[i]*Ei < - self.toler and self.alpha[i] < self.C) :
            nonBound = np.nonzero( np.multiply(self.alpha == 0 ,  self.alpha == self.C ) )[0]
            if len(nonBound > 1):
                maxDiff = -1
                maxJ = -1
                for k in nonBound:
                    if k == i: continue
                    Ek = self.computeLoss( k)
                    if abs(Ei - Ek) > maxDiff:
                        maxDiff = abs(Ei - Ek)
                        maxJ = k
                if self.takeStep(i,maxJ):
                    return 1
               
                j = nonBound[self.selectJrand(i, len(nonBound))]
                if self.takeStep(i,j):
                    return 1
                
            else:
                j = self.selectJrand(i,self.lines)
                if self.takeStep(i,j):
                    return 1
       
        return 0
    
    
    def takeStep(self,i,j):
        if i == j : return 0
        
        Ei = self.computeLoss(i)
        Ej = self.computeLoss(j)
        alpha_i_old = self.alpha[i].copy();
        alpha_j_old = self.alpha[j].copy();
        
        ## compute L,H
        L,H = self.computeLH(i, j);
        if L == H:
            print("L==H")
            return 0
        
        ## compute alpha[j]
        
        eta = self.K[i,i] + self.K[j,j] - 2*self.K[i,j]
        
        if eta <= 0 : 
            print ("K11 + K22 - 2K12 = 0");
            return -1;
        
        alpha_newunc_j = alpha_j_old + float(self.classLabel[j]*(Ei - Ej)/eta);
        
        if alpha_newunc_j < L:
            alpha_newunc_j =  L;
        if alpha_newunc_j > H:
            alpha_newunc_j =  H;
            
    
        ## compute if alpha change enough
      
        if abs(self.alpha[j] - alpha_newunc_j) < self.threshold :
            print(" j has not enough chage ");
            return 0
        
       
        ## update alpha1 
        # a^old(i)yi + a^old(j)yj = τ = a^new(i)yi + a^new(j)yj
       
        self.alpha[i] = alpha_i_old + self.classLabel[j]*self.classLabel[i]*(alpha_j_old - alpha_newunc_j);
        self.alpha[j] = alpha_newunc_j
        self.eCache[i] = Ei
        self.eCache[j] = Ej
        b_i_new = self.b - Ei - self.classLabel[i]*self.K[i,i]*(self.alpha[i] - alpha_i_old) \
            - self.classLabel[j]*self.K[i,j]*(self.alpha[j] - alpha_j_old);
            
        b_j_new = self.b - Ej - self.classLabel[i]*self.K[i,j]*(self.alpha[i] - alpha_i_old) \
            - self.classLabel[j]*self.K[j,j]*(self.alpha[j] - alpha_j_old);
            
        ## updata b
        # 1.  0 < alpha[i] < C,b^new = alpha[i]
        # 2. alpha[i],alpha[j] = 0 or alpha[i],alpha[j] = C, b = (b[i] + b[j])/2
        
        if ( self.alpha[i] > 0 and self.alpha[i] < self.C):
            self.b = b_i_new;
        elif ( self.alpha[j] > 0 and self.alpha[j] < self.C):
            self.b = b_j_new;
        else:
            self.b = (b_j_new + b_i_new)/2.0;
        return 1
        
    def platSmo(self): 
       
        examineAll = True
        numChanged = 0
        iter = 0
        while (iter < self.maxIter) and  ((numChanged > 0) or (examineAll)):
            
            numChanged = 0
            if examineAll:
                for i in range(self.lines): 
                    numChanged += self.selectJ(i)
                    print("full set , iter :%d, i:%d,pairs changes %d" %(iter,i,numChanged)) 
                  
            else:
                nonBound = np.nonzero( np.multiply((self.alpha > 0),(self.alpha < self.C)))[0]
                
                for i in nonBound:
                    numChanged += self.selectJ(i)
                    print("nonBound set , iter :%d, i:%d,pairs changes %d" %(iter,i,numChanged)) 
            
            
            if examineAll:
                examineAll = False
            elif numChanged == 0: 
                examineAll = True
          
            print ("iteration number: %d" % iter)
            iter += 1
          
        return self.alpha,self.b    
    

    def _run(self):
        
        self.platSmo()
        self.weigth = self.calWeigth()
        
'''

path = "../../../data/svm/testSet.txt";
dataIn,classLabel = DataUtil.loadDateFloat(path, float)
smo = SMOMpj(dataIn,classLabel,0.5,0.001  ,100)
smo._run()
print(smo.alpha)
print(smo.b)
'''    