import torch
from torch import nn
class MLP(nn.Module):
    def __init__(self, input_dim,hidden_dim1,hidden_dim2,output_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,hidden_dim1),nn.ReLU(),nn.Dropout(0.3),
                               nn.Linear(hidden_dim1,hidden_dim2),nn.ReLU(),nn.Dropout(0.3),
                               nn.Linear(hidden_dim2,output_dim)
                               )
    def forward(self,x):
        return self.net(x)
    
    def kaiming_init():
        pass
