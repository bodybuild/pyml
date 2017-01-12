'''
Created on 2016年6月29日

Eight queen problem

@author: Ma
'''

class EigthQueen:
    def __init__(self,name):
        self.value = "%s good Job!" % name;
        
    def conflict(self,state, nextX):
        nextY = len(state);
        for i in range(nextY):
            if abs(nextX - state[i]) in (0, abs(nextY - i)) :
                return True;
        return False;
    
    def Queen(self,num,state = ()):
       
        for pos in range(num):
            if not self.conflict(state, pos):
                if len(state) == num - 1 :
                    yield (pos,);
                else:
                    for ll in self.Queen(num, state + (pos,)) :
                        yield (pos,) + ll;
    
    def line(self,pos,length):
        print('* '*(pos) + 'X ' + '* '*(length - pos - 1)) ;
                        
    def prettyPrint(self,seq):
        high = len(seq);
        for pos in range(high):
            self.line(seq[pos],high);
    
    def printValue(self):
        print(self.value);  
    
     
eightQueen = EigthQueen("tony");
result = list(eightQueen.Queen(10));
print(result);
for solution in result:
    print(solution);
    eightQueen.prettyPrint(solution);   
    

eightQueen.printValue();        
                    
                            


            
    


