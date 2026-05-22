import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import copy
class Trainer:
    def __init__(self,net,train_feature,train_label,test_feature,batch_size,lr,num_epochs,weight_decay,optimizer=None,loss=None,device=None):
        self.net=net
        if device is None:
            self.device=torch.device('cuda')
        else:
            self.device=device
        self.net.to(self.device)
        self.train_feature=torch.tensor(train_feature.values,dtype=torch.float32,device=self.device)
        self.train_label=torch.tensor(train_label.values,dtype=torch.long,device=self.device).reshape(-1)
        self.test_feature=torch.tensor(test_feature.values,dtype=torch.float32,device=self.device)
        self.num_epochs=num_epochs
        self.batch_size=batch_size
        self.lr=lr
        self.weight_decay=weight_decay
        if optimizer is None:
            self.optimizer=torch.optim.SGD(self.net.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        else:
            self.optimizer=optimizer(self.net.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        if loss is None:
            self.loss=nn.CrossEntropyLoss()
        else:
            self.loss=loss
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss':[],
            'val_acc':[],
            'test_loss': [],
            'test_acc': []
        }
        self.best_score=0.0
        self.best_epoch=0
        self.best_model_state= None
        self.initial_model_state=copy.deepcopy(self.net.state_dict())
        self.optimizer_class=torch.optim.SGD if optimizer is None else optimizer
        
    def batch_iter(self,X,y=None,shuffle=False):
        n=X.shape[0]
        if shuffle:
            indices=torch.randperm(n,device=X.device)
            for start in range(0,n,self.batch_size):
                batch_idx=indices[start:start+self.batch_size]
                X_batch=X.index_select(0,batch_idx)
                if y is None:
                    yield X_batch
                else:
                    y_batch=y.index_select(0,batch_idx)
                    yield X_batch,y_batch
        else:
            for start in range(0,n,self.batch_size):
                end=min(start+self.batch_size,n)
                X_batch=X[start:end]
                if y is None:
                    yield X_batch
                else:
                    y_batch=y[start:end]
                    yield X_batch,y_batch
        
    def accuracy(self,y_hat,y):
        y_hat=y_hat.argmax(dim=1)
        count=(y_hat.type(y.dtype)==y).sum()
        return count

    def train(self,X_train,y_train,X_valid=None,y_valid=None):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss':[],
            'val_acc':[],
            'test_loss': [],
            'test_acc': []
        }
        train_loss_record,train_acc_record,val_loss_record,val_acc_record=[],[],[],[]
        for epoch in range(self.num_epochs):
            self.net.train()
            metric_loss=0.0
            metric_acc=0
            metric_total=0
            for X,y in self.batch_iter(X_train,y_train,shuffle=True):
                self.optimizer.zero_grad()
                y_hat=self.net(X)
                l=self.loss(y_hat,y)
                l.backward()
                self.optimizer.step()
                with torch.no_grad():
                    batch=X.shape[0]
                    metric_loss+=l.detach()*batch
                    metric_acc+=self.accuracy(y_hat,y)
                    metric_total+=batch
            train_loss=(metric_loss/metric_total).item()
            train_acc=(metric_acc/metric_total).item()
            if X_valid is not None:
                val_loss,val_acc=self.evaluate(X_valid,y_valid)
                val_loss_record.append(val_loss)
                val_acc_record.append(val_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                if val_acc>self.best_score:
                    self.best_score=val_acc
                    self.best_epoch=epoch
                    self.best_model_state=copy.deepcopy(self.net.state_dict())
            train_loss_record.append(train_loss)
            train_acc_record.append(train_acc)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
        if X_valid is not None:
            return train_loss_record,train_acc_record,val_loss_record,val_acc_record
        return train_loss_record,train_acc_record
                
    def evaluate(self,X_valid,y_valid):
        self.net.eval()
        metric_loss=0.0
        metric_acc=0
        metric_total=0
        with torch.no_grad():
            for X,y in self.batch_iter(X_valid,y_valid,shuffle=False):
                y_hat=self.net(X)
                l=self.loss(y_hat,y)
                batch=X.shape[0]
                metric_loss+=l.detach()*batch
                metric_acc+=self.accuracy(y_hat,y)
                metric_total+=batch
            test_loss=(metric_loss/metric_total).item()
            test_acc=(metric_acc/metric_total).item()
        return test_loss,test_acc
    
    def get_k_fold_data(self,k,i):
        fold_size=self.train_feature.shape[0]//k
        X_train,y_train=None,None
        for j in range(k):
            if j==k-1:
                idx=slice(j*fold_size,None)
            else:
                idx=slice(j*fold_size,(j+1)*fold_size)
            if j==i:
                X_valid,y_valid=self.train_feature[idx,:],self.train_label[idx]
            elif X_train is None:
                X_train,y_train=self.train_feature[idx,:],self.train_label[idx]
            else:
                X_train=torch.cat([X_train,self.train_feature[idx,:]],dim=0)
                y_train=torch.cat([y_train,self.train_label[idx]],dim=0)
        return X_train,y_train,X_valid,y_valid
                
    def k_fold_cross_validation(self,k):
        train_acc_sum,val_acc_sum=0.0,0.0
        self.best_score=0.0
        self.best_epoch=0
        self.best_model_state= None
        for i in range(k):
            self.net.load_state_dict(copy.deepcopy(self.initial_model_state))
            self.optimizer=self.optimizer_class(self.net.parameters(),lr=self.lr,weight_decay=self.weight_decay)
            data=self.get_k_fold_data(k,i)
            train_loss,train_acc,val_loss,val_acc=self.train(*data)
            train_acc_sum+=train_acc[-1]
            val_acc_sum+=val_acc[-1]
            print(f'fold {i + 1}, train acc {float(train_acc[-1]):.4f}, '
                  f'validation acc {float(val_acc[-1]):f}')
        print(f'Best validation Accuray: {self.best_score:.4f}, '
              f'Epoch : {self.best_epoch}'
              )
        return train_acc_sum/k,val_acc_sum/k
    
    def best_estimator(self):
        if self.best_model_state is None:
            print('No best model found')
        else:
            self.net.load_state_dict(copy.deepcopy(self.best_model_state))
            self.net.to(self.device)
        return self.net

    def plot(self,val=True):
        epochs=range(1,len(self.history['train_loss'])+1)
        plt.figure(figsize=(8,5))
        plt.plot(epochs,self.history['train_acc'],label='Train ACC')
        if val==True:
            plt.plot(epochs,self.history['val_acc'],label='Val ACC')
        plt.xlabel('Epochs')
        plt.ylabel('ACC')
        plt.legend()
        plt.title('ACC vs Epochs')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(8,5))
        plt.plot(epochs,self.history['train_loss'],label='Train Loss')
        if val==True:
            plt.plot(epochs,self.history['val_loss'],label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss vs Epochs')
        plt.grid(True)
        plt.show()

    def predict(self,test_data):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss':[],
            'val_acc':[],
            'test_loss': [],
            'test_acc': []
        }
        self.net.load_state_dict(copy.deepcopy(self.initial_model_state))
        self.optimizer=self.optimizer_class(self.net.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        #train on whole dataset
        train_loss,train_acc=self.train(self.train_feature,self.train_label)
        net=self.best_estimator()
        net.eval()
        with torch.no_grad():
            prediction=net(self.test_feature.to(self.device)).argmax(dim=1).detach().cpu().numpy()
        test_data['PitNextLap']=pd.Series(prediction.reshape(-1))
        submission=pd.concat([test_data['id'],test_data['PitNextLap']],axis=1)
        submission.to_csv('submission.csv', index=False)