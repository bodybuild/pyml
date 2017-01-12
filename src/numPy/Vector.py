'''
Created on 2016年6月30日

@author: Ma
'''

from numpy import *
class VectorTest:
    
    def numPySum(self,n):
        a = arange(n) ** 2;
        b = arange(n) ** 3;
        c = arange(n);
        print(a);
        print(b);
        print(c);
        c = a + b;
        print(c);
        
vector = VectorTest();
#vector.numPySum(7);

t = dtype([('name',str_,40),("numitems",int32),("price",float32)]);
print(t);
print(t['name']);

item =array([('Meaning of life : DVD',2,10.12),('Fast && Furious 7: DVD',10,10.11),('Kung Fu Panda 3: DVD',2,11.12)] ,dtype = t);

print(item[0][0]);
 
 
mulArray = arange(24).reshape(2,3,4);
print(mulArray);
 
print(mulArray[: ,1,0]);
       
