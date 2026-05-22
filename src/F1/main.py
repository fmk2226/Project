import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Trainer import Trainer
from IPython import display

display.display = lambda *args, **kwargs: None
display.clear_output = lambda *args, **kwargs: None

df_train=pd.read_csv('../../data/F1/train.csv')
df_test=pd.read_csv('../../data/F1/test.csv')

#DATA CLEANING
drop_cols=['id']
df_train_clean=df_train.drop(drop_cols,axis=1)
df_test_clean=df_test.drop(drop_cols,axis=1)
del df_train
del df_test
df_train_copy=df_train_clean.copy() #make a copy of dataframe before removing outliers
df_test_copy=df_test_clean.copy()

#remove outliers
abn_col=['LapTime (s)','LapTime_Delta','Cumulative_Degradation']
upper_bound=df_train_clean[abn_col].quantile(0.999)
lower_bound=df_train_clean[abn_col].quantile(0.001)
lower_bound['LapTime (s)']=df_train_clean['LapTime (s)'].min() #only remove large outliers for LapTime
mask=(
    (df_train_clean[abn_col]>=lower_bound) &
    (df_train_clean[abn_col]<=upper_bound)
).all(axis=1)
df_train_clean=df_train_clean[mask]
df_test_clean[abn_col]=df_test_clean[abn_col].clip(lower=lower_bound,upper=upper_bound,axis=1)

#PREPROCESSING
n_train=df_train_clean.shape[0]
all_features=pd.concat([df_train_clean.iloc[:,:-1],df_test_clean],axis=0)

#one-hot encoding
all_features=pd.get_dummies(all_features,dtype=bool)
numeric_features=all_features.select_dtypes(exclude=[bool]).columns
categorical_features=all_features.select_dtypes(include=[bool]).columns
X_train=all_features.iloc[:n_train]
X_test=all_features.iloc[n_train:]
del all_features

#standardize
stnd=StandardScaler().set_output(transform="pandas")
X_train=pd.concat([stnd.fit_transform(X_train[numeric_features]),X_train[categorical_features].astype(float)],axis=1)
X_test=pd.concat([stnd.transform(X_test[numeric_features]),X_test[categorical_features].astype(float)],axis=1)
y_train=df_train_clean['PitNextLap']

#Hyperparameter
batch_size,lr,num_epochs=128,0.01,32

#Transfer to Data_loader instance
X_train=torch.tensor(X_train.values,dtype=torch.float32)
X_test=torch.tensor(X_test.values,dtype=torch.float32)
y_train=torch.tensor(y_train.values,dtype=torch.float32)
train_iter=DataLoader(
    TensorDataset(X_train,y_train),
    batch_size=batch_size,
    shuffle=True,
)