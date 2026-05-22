import torch
from torch import nn
class MLP(nn.Module):
    def __init__(self, input_dim,hidden_dim1,hidden_dim2,hidden_dim3,output_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,hidden_dim1),nn.ReLU(),nn.Dropout(0.3),
                               nn.Linear(hidden_dim1,hidden_dim2),nn.ReLU(),nn.Dropout(0.3),
                               nn.Linear(hidden_dim2,hidden_dim3),nn.ReLU(),nn.Dropout(0.1),
                               nn.Linear(hidden_dim3,output_dim)
                               )
    def forward(self,x):
        return self.net(x)
    
    @staticmethod
    def kaiming_init(m):
        if isinstance(m,nn.Linear):
            nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    @staticmethod
    def xavier_init(m):
        if isinstance(m,nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

