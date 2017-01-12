'''
Created on 2016年7月22日

@author: Ma
'''
from util.datautil.DataUtil import DataUtil ; 
from machineLearning.svm.SMO import SMO
from machineLearning.svm.svmMLiA import smoSimple,smoP
from machineLearning.svm.SMOMpj import SMOMpj
import time
class SvmPlot:
    
   
    def _run(self):
        
        path = "../../../data/svm/testSetRBF.txt";
        DataUtil.plotScatter(path);
        
        dataIn,classLabel = DataUtil.loadDateFloat(path, float)
        
       
        
        smo_time =  time.time()
        smo = SMOMpj(dataIn,classLabel,0.5,0.001  ,100,method = ("gaussian", 0.4))
        smo._run()
        smo_cost =  time.time() - smo_time
        
        for i in range(len(smo.alpha)):
            if smo.alpha[i] != 0:
                xy = (dataIn[i][0],dataIn[i][1]);
                DataUtil.pltCycle(xy, 0.04);
        
        '''
        path = "../../../data/svm/testSet.txt";
        DataUtil.plotScatter(path);
        
        dataIn,classLabel = DataUtil.loadDateFloat(path, float)
      
        smoP_time = time.time()
        bC,alphaC =  smoP(dataIn, classLabel, 0.5,0.001  ,100)
        smoP_cost = time.time() - smoP_time
        weigth = SMO.returnWeigth(alphaC, bC, path)
        minX,maxX = DataUtil.getXrange(dataIn[:,0])
        DataUtil.pltPlot(weigth, "SVM-c",minX,maxX)
        
        
        
        #CNum = [0.2,0.4,0.6,0.8];
        #for Iter in CNum:
        
        platSmo_time =  time.time()
        alphaO,bO = SMO.platSmo(dataIn,classLabel,0.5,0.001  ,100);
        platSmo_cost =  time.time() - platSmo_time
        weigth = SMO.returnWeigth(alphaO, bO, path);
        minX,maxX = DataUtil.getXrange(dataIn[:,0]);
        DataUtil.pltPlot(weigth, "SVM-o",minX,maxX);
        
        
        smo_time =  time.time()
        smo = SMOMpj(dataIn,classLabel,0.5,0.001  ,100)
        smo._run()
        smo_cost =  time.time() - smo_time
        weigth = SMO.returnWeigth(smo.alpha, smo.b.A.reshape(-1)[0], path);
        minX,maxX = DataUtil.getXrange(dataIn[:,0]);
        DataUtil.pltPlot(weigth, "SVM-n",minX,maxX);
        
        
        print(" time for smoP %f,  time for platSmo %f, time for smo %f " %(smoP_cost,platSmo_cost,smo_cost))
        '''
        '''
       
        simple_currentTime = datetime.datetime.now().microsecond
        alpha,b = SMO.simpleSmo(dataIn,classLabel,0.5,0.001  ,50);
        simple_cost = datetime.datetime.now().microsecond - plat_currentTime
        weigth = SMO.returnWeigth(alpha, b, path);
        DataUtil.pltPlot(weigth, "SVM-simple",minX,maxX);
        
       
      
      
        
                
       
        lg = Logistic();
        weights,weigthX,weigthY = lg.batchGradientDescent(path, 0, 0.001, 1000);
        DataUtil.pltPlot(np.asarray(weights).reshape(-1),"logistic",-2,11);
        '''
        
        DataUtil.plotShow();
        
sp = SvmPlot();
sp._run();