'''
Created on 2016年8月17日

@author: Ma
'''


import numpy as np
import datetime
from util.datautil.DataUtil import DataUtil 
import time

x = [i for i in range(10)]
alpha = np.mat(x).T
print(x)
print(np.nonzero(x))
print(np.nonzero( [i > 1 and i < 5 for i in x]))
print(alpha)
print( np.multiply( (alpha > 1) , (alpha < 5 ) ))
print(np.nonzero( np.multiply( (alpha > 1) , (alpha < 5 ) ) ) )

print(datetime.datetime.now().microsecond)

time.sleep(3)
print(datetime.datetime.now().microsecond)
11111