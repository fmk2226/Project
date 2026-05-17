import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from d2l import torch as d2l
from sklearn.model_selection import train_test_split
from Trainer import Trainer
from IPython import display

display.display = lambda *args, **kwargs: None
display.clear_output = lambda *args, **kwargs: None

data=pd.read_csv('../../data/job/job_search_platform_efficacy_100k.csv')
drop_col=['Student_ID','Time_to_Offer_Days','Offer_Salary','Company_Size_Offered','Accepted_Offer','Role_Relevance']
data_clean=data.drop(drop_col,axis=1)

#doing one-hot encoding and then identify numerical/categorical features
data_one_hot=pd.get_dummies(data_clean,dummy_na=True,dtype=bool)
X=data_one_hot.drop(columns='Offer_Received')
y=data_one_hot['Offer_Received']
numeric_features=X.dtypes[X.dtypes != 'bool'].index
categorical_features=X.dtypes[X.dtypes == 'bool'].index

#split trian/test set and do standardization
train_features,test_features,train_labels,test_labels = train_test_split(X, y, test_size=0.2, random_state=42)
train_mean = train_features[numeric_features].mean()
train_std = train_features[numeric_features].std()

train_features[numeric_features] = (
    train_features[numeric_features] - train_mean
) / train_std

test_features[numeric_features] = (
    test_features[numeric_features] - train_mean
) / train_std

train_features[categorical_features]=train_features[categorical_features].astype(float)
test_features[categorical_features]=test_features[categorical_features].astype(float)
print(train_features.shape)
print(train_labels.shape)
print(test_features.shape)
print(test_labels.shape)

in_features=train_features.shape[1]

# convert DataFrame/Series -> TensorDataset/DataLoader for d2l.train_ch6
train_features_tensor = torch.tensor(train_features.values, dtype=torch.float32)
test_features_tensor = torch.tensor(test_features.values, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels.values, dtype=torch.long)
test_labels_tensor = torch.tensor(test_labels.values, dtype=torch.long)

batch_size = 64
train_iter = DataLoader(
    TensorDataset(train_features_tensor, train_labels_tensor),
    batch_size=batch_size,
    shuffle=True
)
test_iter = DataLoader(
    TensorDataset(test_features_tensor, test_labels_tensor),
    batch_size=batch_size,
    shuffle=False
)

#define model
net=nn.Sequential(nn.Linear(in_features,128),nn.ReLU(),nn.Dropout(0.1),
                  nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.1),
                  nn.Linear(64,2))

def kaiming_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

net.apply(kaiming_init)
lr,num_epochs=0.01,56
trainer=Trainer(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu(),loss=nn.CrossEntropyLoss())
trainer.train()
trainer.plot_history()



