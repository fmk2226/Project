import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import random
from pathlib import Path
from IPython import display
import DataInit
from model import resnet18,xavier_init
from Trainer import Trainer

def RandomSeed(seed:int):
    random.seed(seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed)

RandomSeed(42)
display.display = lambda *args, **kwargs: None
display.clear_output = lambda *args, **kwargs: None
BASE_DIR=Path(__file__).resolve().parents[2]

#Parameters
demo=True
batch_size=32 if demo else 128
valid_ratio=0.1
lr,lr_period,lr_decay=5e-4,4,0.9
num_epochs=20
weight_decay=5e-4
net=resnet18(10,3)
net.apply(xavier_init)

#initialize dataset to a readable form for torchvision
data_dir=DataInit.load_data(BASE_DIR,demo=demo)
DataInit.reorg_cifar10_data(data_dir,valid_ratio)

#data augmentation
transform_train=torchvision.transforms.Compose([
    #resize height and width to 40 pixels
    torchvision.transforms.Resize(40),
    #random crop a square from 0.64-1.0 and then resize to 32 pixels
    torchvision.transforms.RandomResizedCrop(32,scale=(0.64,1.0),ratio=(1.0,1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.4914,0.4822,0.4465],
                                     std=[0.2023,0.1994,0.2010])
])
transform_test=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
])

#load dataset
train_ds,train_valid_ds=[torchvision.datasets.ImageFolder(
    os.path.join(data_dir,'train_valid_test',folder),transform=transform_train)
    for folder in ['train','train_valid']
    ]
valid_ds,test_ds=[torchvision.datasets.ImageFolder(
    os.path.join(data_dir,'train_valid_test',folder),transform=transform_test)
    for folder in ['valid','test']
    ]

train_iter,train_valid_iter=[DataLoader(dataset,batch_size,shuffle=True,drop_last=True)
                             for dataset in (train_ds,train_valid_ds)]
valid_iter=DataLoader(valid_ds,batch_size,shuffle=False,drop_last=False)
test_iter=DataLoader(test_ds,batch_size,shuffle=False,drop_last=False)

trainer=Trainer(net,train_iter,train_valid_iter,valid_iter,test_iter,batch_size,lr,lr_period,lr_decay,num_epochs,weight_decay)
trainer.train()
trainer.plot()