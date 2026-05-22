import torch
from torch import nn
import matplotlib.pyplot as plt
import copy
class Trainer:
    def __init__(self,net,train_iter,test_iter,num_epochs,lr,optimizer=None,loss=None,device=None):
        self.net=net
        self.train_iter=train_iter
        self.test_iter=test_iter
        self.num_epochs=num_epochs
        self.lr=lr
        if optimizer is None:
            self.optimizer=torch.optim.SGD(self.net.parameters(),lr=self.lr)
        else:
            self.optimizer=optimizer(self.net.parameters(),lr=self.lr)
        if loss is None:
            self.loss=nn.CrossEntropyLoss()
        else:
            self.loss=loss
        if device is None:
            self.device=torch.device('cuda')
        else:
            self.device=device
        self.net.to(self.device)
        self.history={
            'train_loss':[],
            'train_mse':[],
            'test_loss':[],
            'test_mse':[]
        }
        self.best_score=float('inf')
        self.best_epoch=0
        self.best_model_state= None
        
    def MSE(self,y_hat,y):
        squared_error=((y_hat-y)**2).sum().item()
        return squared_error

    def train(self):
        for epoch in range(self.num_epochs):
            self.net.train()
            metric_loss=0.0
            metric_mse=0.0
            metric_total=0
            for i,(X,y) in enumerate(self.train_iter):
                X=X.to(self.device)
                y=y.to(self.device)
                self.optimizer.zero_grad()
                y_hat=self.net(X)
                l=self.loss(y_hat,y)
                l.backward()
                self.optimizer.step()
                with torch.no_grad():
                    batch=X.shape[0]
                    metric_loss+=l.item()*batch
                    metric_mse+=self.MSE(y_hat,y)
                    metric_total+=batch
                train_loss=metric_loss/metric_total
                train_mse=metric_mse/metric_total
            test_loss,test_mse=self.evaluate(self.test_iter)
            if test_mse<self.best_score:
                self.best_score=test_mse
                self.best_epoch=epoch+1
                self.best_model_state=copy.deepcopy(self.net.state_dict())
                
            self.history['train_loss'].append(train_loss)
            self.history['train_mse'].append(train_mse)
            self.history['test_loss'].append(test_loss)
            self.history['test_mse'].append(test_mse)
            print(f'epoch: {epoch+1},'
                f'train_loss: {train_loss:.4f},'
                f'train_mse: {train_mse:.4f},'
                f'test_loss: {test_loss:.4f},'
                f'test_mse: {test_mse:.4f}'
            )

    def evaluate(self,data_iter):
        self.net.eval()
        metric_loss=0.0
        metric_mse=0.0
        metric_total=0
        with torch.no_grad():
            for X,y in data_iter:
                X=X.to(self.device)
                y=y.to(self.device)
                y_hat=self.net(X)
                l=self.loss(y_hat,y)
                batch=X.shape[0]
                metric_loss+=l.item()*batch
                metric_mse+=self.MSE(y_hat,y)
                metric_total+=batch
            test_loss=metric_loss/metric_total
            test_mse=metric_mse/metric_total
        return test_loss,test_mse

    def plot(self):
        epoch=range(1,len(self.history['train_loss'])+1)
        plt.figure(figsize=(8,5))
        plt.plot(epoch,self.history['train_mse'],label='Train MSE')
        plt.plot(epoch,self.history['test_mse'],label='Test MSE')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend()
        plt.title('MSE vs Epochs')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(8,5))
        plt.plot(epoch,self.history['train_loss'],label='Train Loss')
        plt.plot(epoch,self.history['test_loss'],label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss vs Epochs')
        plt.grid(True)
        plt.show()

    def best_estimator(self):
        if self.best_model_state is None:
            print("No best model found.")
            return None
        self.net.load_state_dict(self.best_model_state)
        print(
            f'Best Model loaded from epoch {self.best_epoch}, '
            f'best test MSE: {self.best_score:.4f}'
        )