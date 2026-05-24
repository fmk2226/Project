import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import copy
class Trainer:
    def __init__(self,net,train_feature,train_label,test_feature,batch_size,lr,num_epochs,weight_decay,optimizer=None,loss=None,device=None,class_weights=None):
        self.net=net
        if device is None:
            self.device=torch.device('cuda')
        else:
            self.device=device
        if self.device.type=='cuda':
            torch.set_float32_matmul_precision('high')
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
            self.loss=nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss=loss
        self.loss.to(self.device)
        self.history = {
            'train_loss': [],
            'train_f1': [],
            'val_loss':[],
            'val_f1':[],
            'val_threshold':[],
            'test_loss': [],
            'test_f1': []
        }
        self.best_score=0.0
        self.best_epoch=0
        self.best_threshold=0.5
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

    def f1(self,y_hat,y,threshold=0.5):
        prob=torch.softmax(y_hat,dim=1)[:,1]
        return self.f1_score_from_prob(prob,y,threshold=threshold)

    def best_f1_threshold(self,prob,y):
        thresholds=torch.arange(5,96,device=prob.device,dtype=prob.dtype)/100
        pred=prob.unsqueeze(0)>=thresholds.unsqueeze(1)
        label=y.bool().unsqueeze(0)
        tp=(pred & label).sum(dim=1).float()
        fp=(pred & ~label).sum(dim=1).float()
        fn=(~pred & label).sum(dim=1).float()
        denom=2*tp+fp+fn
        scores=torch.where(denom>0,2*tp/denom,torch.zeros_like(denom))
        best_idx=torch.argmax(scores)
        return scores[best_idx].item(),thresholds[best_idx].item()

    def f1_counts_from_prob(self,prob,y,threshold=0.5):
        pred=prob>=threshold
        label=y.bool()
        tp=(pred & label).sum().float()
        fp=(pred & ~label).sum().float()
        fn=(~pred & label).sum().float()
        return tp,fp,fn

    def f1_from_counts(self,tp,fp,fn):
        denom=2*tp+fp+fn
        if denom.item()==0:
            return 0.0
        return (2*tp/denom).item()
        
    def train(self,X_train,y_train,X_valid=None,y_valid=None):
        self.history = {
            'train_loss': [],
            'train_f1': [],
            'val_loss':[],
            'val_f1':[],
            'val_threshold':[],
            'test_loss': [],
            'test_f1': []
        }
        train_loss_record,train_f1_record,val_loss_record,val_f1_record=[],[],[],[]
        for epoch in range(self.num_epochs):
            self.net.train()
            metric_loss=0.0
            metric_total=0
            train_tp=torch.tensor(0.0,device=self.device)
            train_fp=torch.tensor(0.0,device=self.device)
            train_fn=torch.tensor(0.0,device=self.device)
            for X,y in self.batch_iter(X_train,y_train,shuffle=True):
                self.optimizer.zero_grad(set_to_none=True)
                y_hat=self.net(X)
                l=self.loss(y_hat,y)
                l.backward()
                self.optimizer.step()
                with torch.no_grad():
                    batch=X.shape[0]
                    metric_loss+=l.detach()*batch
                    metric_total+=batch
                    prob=torch.softmax(y_hat,dim=1)[:,1].detach()
                    tp,fp,fn=self.f1_counts_from_prob(prob,y,threshold=0.5)
                    train_tp+=tp
                    train_fp+=fp
                    train_fn+=fn
            train_loss=(metric_loss/metric_total).item()
            train_f1=self.f1_from_counts(train_tp,train_fp,train_fn)
            if X_valid is not None:
                val_loss,val_f1,val_threshold=self.evaluate(X_valid,y_valid,search_threshold=True)
                val_loss_record.append(val_loss)
                val_f1_record.append(val_f1)
                self.history['val_loss'].append(val_loss)
                self.history['val_f1'].append(val_f1)
                self.history['val_threshold'].append(val_threshold)
                if val_f1>self.best_score:
                    self.best_score=val_f1
                    self.best_epoch=epoch
                    self.best_threshold=val_threshold
                    self.best_model_state=copy.deepcopy(self.net.state_dict())
            train_loss_record.append(train_loss)
            train_f1_record.append(train_f1)
            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_f1)
        if X_valid is not None:
            return train_loss_record,train_f1_record,val_loss_record,val_f1_record
        return train_loss_record,train_f1_record

    def f1_score_from_prob(self,prob,y,threshold=0.5):
        tp,fp,fn=self.f1_counts_from_prob(prob,y,threshold=threshold)
        return self.f1_from_counts(tp,fp,fn)
                
    def evaluate(self,X_valid,y_valid,search_threshold=False,threshold=0.5):
        self.net.eval()
        metric_loss=0.0
        metric_total=0
        all_prob=[]
        all_label=[]
        with torch.no_grad():
            for X,y in self.batch_iter(X_valid,y_valid,shuffle=False):
                y_hat=self.net(X)
                l=self.loss(y_hat,y)
                batch=X.shape[0]
                metric_loss+=l.detach()*batch
                metric_total+=batch
                all_prob.append(torch.softmax(y_hat,dim=1)[:,1].detach())
                all_label.append(y.detach())
            test_loss=(metric_loss/metric_total).item()
            prob=torch.cat(all_prob)
            label=torch.cat(all_label)
            if search_threshold:
                test_f1,best_threshold=self.best_f1_threshold(prob,label)
            else:
                best_threshold=threshold
                test_f1=self.f1_score_from_prob(prob,label,threshold=threshold)
        if search_threshold:
            return test_loss,test_f1,best_threshold
        return test_loss,test_f1
    
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
        train_f1_sum,val_f1_sum=0.0,0.0
        self.best_score=0.0
        self.best_epoch=0
        self.best_threshold=0.5
        self.best_model_state= None
        for i in range(k):
            self.net.load_state_dict(copy.deepcopy(self.initial_model_state))
            self.optimizer=self.optimizer_class(self.net.parameters(),lr=self.lr,weight_decay=self.weight_decay)
            data=self.get_k_fold_data(k,i)
            train_loss,train_f1,val_loss,val_f1=self.train(*data)
            train_f1_sum+=train_f1[-1]
            val_f1_sum+=val_f1[-1]
            print(f'fold {i + 1}, train f1 {float(train_f1[-1]):.4f}, '
                  f'validation f1 {float(val_f1[-1]):.4f}, '
                  f'threshold {self.history["val_threshold"][-1]:.2f}')
        print(f'Best validation F1: {self.best_score:.4f}, '
              f'Epoch : {self.best_epoch}, '
              f'Threshold : {self.best_threshold:.2f}'
              )
        return train_f1_sum/k,val_f1_sum/k
    
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
        plt.plot(epochs,self.history['train_f1'],label='Train F1')
        if val==True:
            plt.plot(epochs,self.history['val_f1'],label='Val F1')
        plt.xlabel('Epochs')
        plt.ylabel('F1')
        plt.legend()
        plt.title('F1 vs Epochs')
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

    def predict(self,test_data,load_model=False,threshold=None):
        if threshold is None:
            threshold=self.best_threshold
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must be between 0 and 1')
        self.history = {
            'train_loss': [],
            'train_f1': [],
            'val_loss':[],
            'val_f1':[],
            'val_threshold':[],
            'test_loss': [],
            'test_f1': []
        }
        if load_model:
            net=self.best_estimator()
        else:
            self.net.load_state_dict(copy.deepcopy(self.initial_model_state))
            self.optimizer=self.optimizer_class(self.net.parameters(),lr=self.lr,weight_decay=self.weight_decay)
            train_loss,train_f1=self.train(self.train_feature,self.train_label)
            net=self.net
        net.eval()
        with torch.no_grad():
            logits=net(self.test_feature)
            prob=torch.softmax(logits,dim=1)[:,1]
            prediction=(prob>=threshold).long().detach().cpu().numpy()
        test_data['PitNextLap']=pd.Series(prediction.reshape(-1))
        submission=pd.concat([test_data['id'],test_data['PitNextLap']],axis=1)
        submission.to_csv('submission.csv', index=False)