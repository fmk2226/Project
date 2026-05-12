from cProfile import label

import torch
from torch import nn
import matplotlib.pyplot as plt
class Trainer:
    def __init__(self,net,train_iter,test_iter,num_epochs,lr,device=None,optimizer=torch.optim.SGD,loss=None):
        self.net = net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs
        self.lr = lr
        if device is None:
            self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device=device
        self.net.to(self.device)
        self.optimizer=optimizer(self.net.parameters(),lr=self.lr)
        if loss is None:
            self.loss=nn.CrossEntropyLoss()
        else:
            self.loss=loss
        self.history={
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
        }

    def accuracy(self,y_hat,y):
        y_hat=y_hat.argmax(dim=1)
        count=(y_hat.type(y.dtype)==y).sum().item()
        return count

    def train(self):
        for epoch in range(self.num_epochs):
            self.net.train()
            metric_loss=0.0
            metric_correct=0
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
                    batch_size=X.shape[0]
                    metric_loss+=l.item()*batch_size
                    metric_correct+=self.accuracy(y_hat,y)
                    metric_total+=batch_size

                train_loss=metric_loss/metric_total
                train_acc=metric_correct/metric_total
            test_loss,test_acc=self.evaluate(self.test_iter)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            print(f"epoch {epoch+1}, "
                  f"train_loss: {train_loss:.4f}, "
                  f"train_acc: {train_acc:.4f}, "
                  f"test_loss: {test_loss:.4f}, "
                  f"test_acc: {test_acc:.4f}")

    def evaluate(self,data_iter):
        self.net.eval()
        metric_loss=0.0
        metric_correct=0
        metric_total=0
        with torch.no_grad():
            for (X,y) in data_iter:
                X=X.to(self.device)
                y=y.to(self.device)
                y_hat=self.net(X)
                l=self.loss(y_hat,y)
                batch_size=X.shape[0]
                metric_loss+=l.item()*batch_size
                metric_correct+=self.accuracy(y_hat,y)
                metric_total+=batch_size
        avg_loss=metric_loss/metric_total
        avg_acc=metric_correct/metric_total
        return avg_loss,avg_acc
    def plot_history(self):
        epochs=range(1,len(self.history['train_loss'])+1)
        #epochs=range(1,self.num_epochs+1)
        plt.figure(figsize=(8,5))
        plt.plot(epochs,self.history['train_acc'],label='train_acc')
        plt.plot(epochs,self.history['test_acc'],label='test_acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy vs Epochs')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(8,5))
        plt.plot(epochs,self.history['train_loss'],label='train_loss')
        plt.plot(epochs,self.history['test_loss'],label='test_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss vs Epochs')
        plt.grid(True)
        plt.show()