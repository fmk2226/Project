import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from d2l import torch as d2l
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Trainer import Trainer
from IPython import display

display.display = lambda *args, **kwargs: None
display.clear_output = lambda *args, **kwargs: None

df=pd.read_csv('H:/deeplearning/Project/JobProject/data/job_search_platform_efficacy_100k.csv')
drop_col=['Student_ID','Time_to_Offer_Days','Offer_Salary','Company_Size_Offered','Accepted_Offer','Role_Relevance']
df_clean=df.drop(drop_col,axis=1)

def create_target(row):
    if row['Offer_Received'] == 1:
        return 3  # Got Offer
    elif row['Second_Round_Interviews'] > 0:
        return 2  # Second Round Interview but No Offer
    elif row['First_Round_Interviews'] > 0:
        return 1  # Interview but No Offer
    else:
        return 0  # No Interview
    
def return_target(num):
    if num == 3:
        return 'Got Offer'
    elif num == 2:
        return 'Second Round Interview but No Offer'
    elif num == 1:
        return 'Interview but No Offer'
    else:
        return 'No Interview'

target_column = ['First_Round_Interviews','Second_Round_Interviews','Offer_Received']
df_clean['application_result'] = df_clean.apply(create_target, axis=1)
df_clean.drop(columns=target_column, inplace=True)

y = df_clean['application_result']
X = df_clean.drop(columns=['application_result'])

X_numeric = X.select_dtypes(include=[np.number])
X_numeric_columns = X_numeric.columns
X_categorical = X.select_dtypes(include=['object', 'string'])
X_categorical_encoded = pd.get_dummies(X_categorical, dtype=np.float32)

X_temp = pd.concat([X_numeric, X_categorical_encoded], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.2, stratify=y, random_state=0)

scaler = StandardScaler()

X_train_numeric_scaled = scaler.fit_transform(X_train[X_numeric_columns])
X_test_numeric_scaled = scaler.transform(X_test[X_numeric_columns])

X_train_categorical = X_train.drop(columns=X_numeric_columns)
X_test_categorical = X_test.drop(columns=X_numeric_columns)

X_train = pd.concat([
    pd.DataFrame(X_train_numeric_scaled, columns=X_numeric_columns),
    X_train_categorical.reset_index(drop=True)
], axis=1)

X_test = pd.concat([
    pd.DataFrame(X_test_numeric_scaled, columns=X_numeric_columns),
    X_test_categorical.reset_index(drop=True)
], axis=1)

train_features=X_train
train_labels=y_train
test_features=X_test
test_labels=y_test

in_features=train_features.shape[1]
num_classes=y.nunique()

train_features_tensor = torch.tensor(train_features.to_numpy(dtype=np.float32), dtype=torch.float32)
test_features_tensor = torch.tensor(test_features.to_numpy(dtype=np.float32), dtype=torch.float32)
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
                  nn.Linear(64,num_classes))

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
