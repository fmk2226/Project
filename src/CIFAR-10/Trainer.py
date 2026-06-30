import torch
from torch import nn
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self,net,train_iter,train_valid_iter,valid_iter,test_iter,
                 batch_size,lr,lr_period,lr_decay,num_epochs,weight_decay,optimizer=None,
                 loss=None,devices=None,pretrained=True):
        if devices is None:
            self.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        else:
            self.devices = devices
        net=net.to(self.devices[0])
        self.net=nn.DataParallel(net,device_ids=self.devices) if len(self.devices)>1 else net
        self.train_iter=train_iter
        self.train_valid_iter=train_valid_iter
        self.valid_iter=valid_iter
        self.test_iter=test_iter
        self.batch_size=batch_size
        self.lr=lr
        self.lr_period=lr_period
        self.lr_decay=lr_decay
        self.num_epochs=num_epochs
        self.weight_decay=weight_decay
        self.pretrained=pretrained
        if optimizer is None:
            self.optimizer=torch.optim.AdamW
        else:
            self.optimizer=optimizer
        self.scheduler=None
        if loss is None:
            self.loss=nn.CrossEntropyLoss()
        else:
            self.loss=loss()
        self.loss.to(self.devices[0])
        self.history = {
            'train_loss':[],
            'train_acc':[],
            'train_valid_loss':[],
            'train_valid_acc':[],
            'val_loss':[],
            'val_acc':[],
            'test_loss':[],
            'test_acc':[]
        }

    def accuracy(self,y_hat,y):
        y_hat=y_hat.argmax(dim=1)
        count=(y_hat.type(y.dtype)==y).sum()
        return count

    def init_optimizer(self):
        if self.pretrained==True:
            param_1x=[param for name,param in self.net.named_parameters()
                      if name not in ['fc.weight','fc.bias']]
            self.optimizer=self.optimizer([{'params':param_1x},
                                           {'params':self.net.fc.parameters(),'lr':self.lr*10}],
                                           lr=self.lr,weight_decay=self.weight_decay)
        else:
            self.optimizer=self.optimizer(self.net.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,self.lr_period,self.lr_decay)

    def train_epoch(self,train_iter,valid_iter=None):
        self.init_optimizer()
        for epoch in range(self.num_epochs):
            self.net.train()
            metric_loss=0.0
            metric_acc=0
            metric_total=0
            for i,(X,y) in enumerate(train_iter):
                X=X.to(self.devices[0],non_blocking=True)
                y=y.to(self.devices[0],non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                y_hat=self.net(X)
                l=self.loss(y_hat,y)
                l.backward()
                self.optimizer.step()
                with torch.no_grad():
                    batch=X.shape[0]
                    metric_acc+=self.accuracy(y_hat,y)
                    metric_loss+=l.detach()*batch
                    metric_total+=batch
            train_loss=(metric_loss/metric_total).item()
            train_acc=(metric_acc/metric_total).item()
            if valid_iter is not None:
                val_loss,val_acc=self.evaluate(valid_iter)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            print(f"epoch {epoch+1}, "
                  f"train_loss: {train_loss:.4f}, "
                  f"train_acc: {train_acc:.4f}, "
                  f"val_loss: {val_loss:.4f}, "
                  f"val_acc: {val_acc:.4f}")
            self.scheduler.step()

    def evaluate(self,valid_iter):
        self.net.eval()
        metric_loss=0.0
        metric_acc=0
        metric_total=0
        with torch.no_grad():
            for i,(X,y) in enumerate(valid_iter):
                X=X.to(self.devices[0],non_blocking=True)
                y=y.to(self.devices[0],non_blocking=True)
                y_hat=self.net(X)
                l=self.loss(y_hat,y)
                batch=X.shape[0]
                metric_acc+=self.accuracy(y_hat,y)
                metric_loss+=l.detach()*batch
                metric_total+=batch
        val_loss=(metric_loss/metric_total).item()
        val_acc=(metric_acc/metric_total).item()
        return val_loss,val_acc

    def train(self):
        self.train_epoch(self.train_iter,self.valid_iter)

    def predict(self):
        pass

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
