'''
Created on 2016年7月22日

@author: Ma
'''
import numpy as np;
from util.datautil.DataUtil import DataUtil ;
class SMO:
    
    @staticmethod
    def Gx(line,alphas,dataMat,labelMat,b):
        Gx = float((np.multiply(alphas,labelMat)).T*dataMat*(dataMat[line].T) + b);# g(x)
        return Gx;
    
    @staticmethod
    def computeLoss(Gx,label):
        return Gx - float(label);
    
    @staticmethod
    def selectJrand(i,m):
        j = i;
        while j == i:
            j = int(np.random.uniform(0,m));
        return j;
    
    #@staticmethod
    #def selectJ(i,Ei,Elist):
        
    
    @staticmethod
    def computeLH(labelVec,alpha,i,j,C):
        
        if labelVec[i] == labelVec[j]:
            L = max(0,alpha[i] + alpha[j] - C);
            H = min(C,alpha[i] + alpha[j]);
        else:
            L = max(0,alpha[j] - alpha[i]);
            H = min(C, C + alpha[j] - alpha[i]);
        return L,H;
    @staticmethod
    def upateAlphaJ(L,H,dataMat,label,E1,E2,i,j,alpha_old_j):
        eta = dataMat[i]*dataMat[i].T + dataMat[j]*dataMat[j].T + 2*dataMat[i]*dataMat[j].T;
        if eta <= 0 : 
            print ("K11 + K22 - 2K12 = 0");
            return -1;
        alpha_newunc_j = alpha_old_j + float(label*(E1 - E2)/eta);
        
        if alpha_newunc_j < L:
            return L;
        if alpha_newunc_j > H:
            return H;
        return alpha_newunc_j;
    
        
        
    @staticmethod
    def simpleSmo(dataIn,classLabel,C,totler,maxIter,threshold = 0.001):
        
        dataInMat = np.mat(dataIn,np.float);
        
        labelMat = np.mat(classLabel, np.float).T;
        
        lines,cols = np.shape(dataInMat);
        alpha = np.zeros((lines,1));
        b = 0;
        for numOfInter in range( maxIter):
            alphaPairsChanged = 0;
            for i in range(lines):
                Ei = SMO.computeLoss(SMO.Gx(i, alpha, dataInMat, labelMat, b), labelMat[i]) ;
                
                ###check and pick up alpha whitch violate KKT condition
                ## KKT condition
                # 1. alpha = 0 && y(i)*g(i) >= 1,
                # 2. 0< alpha < C && y(i)*g(i) = 1(on the boundray)
                # 3. alpha = C && y(i)*g(i) <=1
                
                ## violate condition
                # 1. y(i)*g(i) >=1 ,alpha >0
                # 2. y(i)*g(i) <=1, alpha <C
                
                # y(i)*E(i) = (g(i) - y(i))y(i) = g(i)*y(i) - 1,
                
                ## So, 
                # 1. y(i)*E(i)  >= 1-1 = 0,alpha > 0
                # 2. y(i)*E(i) <= 1- 1 = 0 ,alpha < C
                
                
                if (labelMat[i]*Ei > totler and alpha[i] > 0) \
                    or (labelMat[i]*Ei < - totler and alpha[i] < C) :
                    
                    ### select second alpha
                    
                    j = SMO.selectJrand(i, lines);  
            
                    Ej = SMO.computeLoss(SMO.Gx(j, alpha, dataInMat, labelMat, b), labelMat[j]);
                
                    alpha_i_old = alpha[i].copy();
                    alpha_j_old = alpha[j].copy();
                    
                    ## compute L,H
                    L,H = SMO.computeLH(labelMat, alpha, i, j, C);
                    if L == H:
                        print("L==H")
                        continue;
                    ## compute alpha[j]
                    alpha[j] = SMO.upateAlphaJ(L, H, dataInMat , labelMat[j], Ei, Ej, i, j, alpha_j_old);
                    if alpha[j] == -1:
                        continue;
                    
                    
                    ## compute if alpha change enough
                    
                    if abs(alpha[j] - alpha_j_old) < threshold :
                        print(" j has not enough change ");
                        continue;
                    
                    ## update alpha1 
                    # a^old(i)yi + a^old(j)yj = τ = a^new(i)yi + a^new(j)yj
                    
                    alpha[i] = alpha_i_old + labelMat[j]*labelMat[i]*(alpha_j_old - alpha[j]);
                    
                    b_i_new = b - Ei - labelMat[i]*(dataInMat[i]*dataInMat[i].T)*(alpha[i] - alpha_i_old) \
                        - labelMat[j]*(dataInMat[i]*dataInMat[j].T)*(alpha[j] - alpha_j_old);
                        
                    b_j_new = b - Ej - labelMat[i]*(dataInMat[i]*dataInMat[j].T)*(alpha[i] - alpha_i_old) \
                        - labelMat[j]*(dataInMat[i]*dataInMat[i].T)*(alpha[j] - alpha_j_old);
                        
                    ## updata b
                    # 1.  0 < alpha[i] < C,b^new = alpha[i]
                    # 2. alpha[i],alpha[j] = 0 or alpha[i],alpha[j] = C, b = (b[i] + b[j])/2
                    
                    if ( alpha[i] > 0 and alpha[i] < C):
                        b = b_i_new;
                    elif ( alpha[j] > 0 and alpha[j] < C):
                        b = b_j_new;
                    else:
                        b = (b_j_new + b_i_new)/2.0;
                    alphaPairsChanged += 1
                    print(b)
                    print(" Iter : %d, i : %d ,j %d, pairs changed %d " %(numOfInter,i,j,alphaPairsChanged));
                    
                   
        return alpha,np.asarray(b).reshape(-1)[0]; 
    
    @staticmethod
    def selectJ(i,Ei,alpha, dataInMat, labelMat, b,lines,costE):
        
        
        maxK = -1; maxDeltaE = 0; Ej = 0
        costE[i] = [1,Ei]
        
        validEcacheList = np.nonzero(costE[:,0])[0]
        if (len(validEcacheList)) > 1:
            for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
                if k == i: continue #don't calc for i, waste of time
                Ek = SMO.computeLoss(SMO.Gx(k, alpha, dataInMat, labelMat, b), labelMat[k]);
                deltaE = abs(Ei - Ek)
                if (deltaE > maxDeltaE):
                    maxK = k; maxDeltaE = deltaE; Ej = Ek
            return maxK, Ej
        else:   #in this case (first time around) we don't have any valid eCache values
            j = SMO.selectJrand(i, lines)
            Ej = SMO.computeLoss(SMO.Gx(j, alpha, dataInMat, labelMat, b), labelMat[j]);
        return j, Ej
        
        '''
        for j in range(lines):
            if i != j:
                
                diff = abs(Ei - Ej)
                if maxDiff <= diff:
                    maxDiff = diff
                    maxJ = j
                    maxEj = Ej
        return maxJ,maxEj
        '''
    @staticmethod
    def innerLoop(i,dataInMat,labelMat,alpha,b,C,totler,threshold,lines,costE):
      
        Ei = SMO.computeLoss(SMO.Gx(i, alpha, dataInMat, labelMat, b), labelMat[i])
       
        ###check and pick up alpha whitch violate KKT condition
        ## KKT condition
        # 1. alpha = 0 && y(i)*g(i) >= 1,
        # 2. 0< alpha < C && y(i)*g(i) = 1(on the boundray)
        # 3. alpha = C && y(i)*g(i) <=1
        
        ## violate condition
        # 1. y(i)*g(i) >=1 ,alpha >0
        # 2. y(i)*g(i) <=1, alpha <C
        
        # y(i)*E(i) = (g(i) - y(i))y(i) = g(i)*y(i) - 1
        
        ## So, 
        # 1. y(i)*E(i)  >= 1-1 = 0,alpha > 0
        # 2. y(i)*E(i) <= 1- 1 = 0 ,alpha < C
        
                
        if (labelMat[i]*Ei > totler and alpha[i] > 0) \
            or (labelMat[i]*Ei < - totler and alpha[i] < C) :
            
            ### select second alpha
            
            j,Ej = SMO.selectJ(i,Ei,alpha, dataInMat, labelMat, b,lines,costE)
           
            
            alpha_i_old = alpha[i].copy();
            alpha_j_old = alpha[j].copy();
            
            ## compute L,H
            L,H = SMO.computeLH(labelMat, alpha, i, j, C);
            if L == H:
                #print("L==H")
                return 0,b
            ## compute alpha[j]
            alpha[j] = SMO.upateAlphaJ(L, H, dataInMat , labelMat[j], Ei, Ej, i, j, alpha_j_old);
            if alpha[j] == -1:
                #print(" alpha_j_new compute Error!")
                return 0,b
            
            costE[j] = [1,Ej]
            ## compute if alpha change enough
            
            if abs(alpha[j] - alpha_j_old) < threshold :
                #print(" j has not enough chage ");
                return 0,b
            costE[i] = [1,Ei]
            ## update alpha1 
            # a^old(i)yi + a^old(j)yj = τ = a^new(i)yi + a^new(j)yj
           
            alpha[i] = alpha_i_old + labelMat[j]*labelMat[i]*(alpha_j_old - alpha[j]);
           
            b_i_new = b - Ei - labelMat[i]*(dataInMat[i]*dataInMat[i].T)*(alpha[i] - alpha_i_old) \
                - labelMat[j]*(dataInMat[i]*dataInMat[j].T)*(alpha[j] - alpha_j_old);
                
            b_j_new = b - Ej - labelMat[i]*(dataInMat[i]*dataInMat[j].T)*(alpha[i] - alpha_i_old) \
                - labelMat[j]*(dataInMat[i]*dataInMat[i].T)*(alpha[j] - alpha_j_old);
                
            ## updata b
            # 1.  0 < alpha[i] < C,b^new = alpha[i]
            # 2. alpha[i],alpha[j] = 0 or alpha[i],alpha[j] = C, b = (b[i] + b[j])/2
            
            if ( alpha[i] > 0 and alpha[i] < C):
                b = b_i_new;
            elif ( alpha[j] > 0 and alpha[j] < C):
                b = b_j_new;
            else:
                b = (b_j_new + b_i_new)/2.0;
            
            return 1,np.asarray(b).reshape(-1)[0]
       
        else:
            return 0,b
    @staticmethod
    def platSmo(dataIn,classLabel,C,totler,maxIter,threshold = 0.00001): 
        dataInMat = np.mat(dataIn)
        labelMat = np.mat(classLabel).T
        lines,cols = np.shape(dataInMat)
       
        
        alpha = np.zeros((lines,1))
        b = 0.0;
       
        completeDataSet = True
        alphaPairsChanges = 0
        
        costE = np.zeros((lines,2))
        iter = 0
        while (iter < maxIter) and  ((alphaPairsChanges > 0) or (completeDataSet)):
            
            alphaPairsChanges = 0
            if completeDataSet:
                for i in range(lines):
                    alphaPairsChangesTmp,b = SMO.innerLoop(i,dataInMat,labelMat,alpha,b,C,totler,threshold,lines,costE)
                    alphaPairsChanges += alphaPairsChangesTmp
                    print("full set , iter :%d, i:%d,pairs changes %d" %(iter,i,alphaPairsChanges)) 
                  
            else:
                nonBound = np.nonzero((alpha > 0)*(alpha < C))[0]
                for i in nonBound:
                    alphaPairsChangesTmp,b = SMO.innerLoop(i,dataInMat,labelMat,alpha,b,C,totler,threshold,lines,costE)
                    alphaPairsChanges += alphaPairsChangesTmp
                    print("nonBound set , iter :%d, i:%d,pairs changes %d" %(iter,i,alphaPairsChanges)) 
            
            
            if completeDataSet:
                completeDataSet = False
            elif alphaPairsChanges == 0: 
                completeDataSet = True
          
            print ("iteration number: %d" % iter)
            iter += 1
          
        return alpha,b   
    @staticmethod
    def returnWeigth(alpha,b,path):
        dataIn,classLabel = DataUtil.loadDateFloat(path, float)
        classMat = np.mat(classLabel).T;
        dataInMat = np.mat(dataIn);
        weigth = (np.multiply(alpha,classMat).T)*dataInMat;

        weigthList = (np.asarray(weigth).reshape(-1));
        return [b,weigthList[0],weigthList[1]];
    def _run(self):
        path = "../../../data/svm/testSet.txt";
        dataIn,classLabel = DataUtil.loadDateFloat(path, float)
        SMO.platSmo(dataIn,classLabel,0.6,0.001  ,10)
        ''' 
        alpha,b = self.simpleSmo(dataIn,classLabel,0.6,0.001  ,10);
       
        print(classLabel);
        classMat = np.mat(classLabel).T;
        dataInMat = np.mat(dataIn);
        weigth = (np.multiply(alpha,classMat).T)*dataInMat;
        print(alpha);
        print(list(np.asarray(weigth).reshape(-1))[0]);
        
        '''
        
#smo = SMO();
#smo._run();

