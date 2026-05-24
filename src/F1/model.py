import torch
from torch import nn

def kaiming_init(m):
    if isinstance(m,nn.Linear):
        nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def xavier_init(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class MLP(nn.Module):
    def __init__(self, input_dim,hidden_dim1,hidden_dim2,output_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,hidden_dim1),nn.BatchNorm1d(hidden_dim1),nn.ReLU(),nn.Dropout(0.3),
                               nn.Linear(hidden_dim1,hidden_dim2),nn.BatchNorm1d(hidden_dim2),nn.ReLU(),nn.Dropout(0.3),
                               nn.Linear(hidden_dim2,output_dim)
                               )
    def forward(self,x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self,hidden_dim,dropout):
        super().__init__()
        self.Linear1=nn.Linear(hidden_dim,hidden_dim)
        self.Linear2=nn.Linear(hidden_dim,hidden_dim)
        self.activation=nn.ReLU()
        self.dropout=nn.Dropout(dropout)
        self.batchnorm=nn.BatchNorm1d(hidden_dim)
    def forward(self,x):
        residual=x
        out=self.dropout(self.activation(self.Linear1(x)))
        out=self.Linear2(out)
        out=out+residual
        out=self.batchnorm(out)
        out=self.activation(out)
        return out

class ResidualRegressor(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,dropout):
        super().__init__()
        self.input_layer=nn.Linear(input_dim,hidden_dim)
        self.activation=nn.ReLU()
        self.dropout=nn.Dropout(dropout)
        self.block1=ResidualBlock(hidden_dim,dropout)
        self.block2=ResidualBlock(hidden_dim,dropout)
        self.block3=ResidualBlock(hidden_dim,dropout)
        self.output_layer=nn.Linear(hidden_dim,output_dim)
    def forward(self,x):
        x=self.input_layer(x)
        x=self.activation(x)
        x=self.dropout(x)
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.output_layer(x)
        return x