import torch
import torch.nn as nn
class test(nn.Module):
    def __init__(self, input):
        super(test,self).__init__()
        self.input = input

    def forward(self,x, a_bool):
        if a_bool:
            return self.input * x, 1000
        else:
            return self.input * x, 2000

T = test(torch.tensor([8,10]))
print(T(6, False))