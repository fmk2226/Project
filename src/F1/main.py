import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import MLP
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
train_used=df_train_clean
test_used=df_test_clean
n_train=train_used.shape[0]
all_features=pd.concat([train_used.iloc[:,:-1],test_used],axis=0)

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
y_train=train_used['PitNextLap']

#HYPERPARAMETER
k,batch_size,lr,num_epochs,weight_decay=10,8192,1e-3,20,1e-4

#MODEL
net=MLP(X_train.shape[1],512,256,2)
net.apply(MLP.kaiming_init)
trainer=Trainer(net,X_train,y_train,X_test,batch_size,lr,num_epochs,weight_decay,optimizer=torch.optim.AdamW)
trainer.k_fold_cross_validation(k)
trainer.plot()
trainer.predict(df_test,load_model=True)
trainer.plot(val=False)