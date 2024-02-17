import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        
        self.l1 = nn.Linear(10,20)
        self.l2 = nn.Linear(20,30)
        
    def forward(self, x):
        # return self.l2(self.l1(x))
        return self.l1(self.l2(x))
    
model = NN()

print('mod')
breakpoint()
