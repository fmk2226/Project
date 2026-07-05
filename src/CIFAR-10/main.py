import os
import torch
import torchvision
from torch.utils.data import DataLoader
import random
from pathlib import Path
from IPython import display
import DataInit
from model import resnet18,pretrained_resnet18
from Trainer import Trainer

def RandomSeed(seed:int):
    random.seed(seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed)

def main():
    RandomSeed(42)
    display.display = lambda *args, **kwargs: None
    display.clear_output = lambda *args, **kwargs: None
    BASE_DIR=Path(__file__).resolve().parents[2]
    
    #Parameters
    demo=True
    pretrained=True
    batch_size=32 if demo else 128
    valid_ratio=0.1
    lr,lr_period,lr_decay=5e-4,4,0.9
    num_epochs=7 if pretrained else 20
    weight_decay=5e-4
    net=pretrained_resnet18() if pretrained else resnet18(10,3)

    #initialize dataset to a readable form for torchvision
    data_dir=DataInit.load_data(BASE_DIR,demo=demo)
    DataInit.reorg_cifar10_data(data_dir,valid_ratio)

    #data augmentation
    normalize=torchvision.transforms.Normalize(
        mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]
        ) if pretrained else torchvision.transforms.Normalize(
        mean=[0.4914,0.4822,0.4465],std=[0.2023,0.1994,0.2010]
        )
    transform_train=torchvision.transforms.Compose([
        #resize height and width to 40 pixels
        torchvision.transforms.Resize(40),
        #random crop a square from 0.64-1.0 and then resize to 32 pixels
        torchvision.transforms.RandomResizedCrop(32,scale=(0.64,1.0),ratio=(1.0,1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize
    ])
    transform_test=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalize
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

    train_iter,train_valid_iter=[DataLoader(dataset,batch_size,shuffle=True,drop_last=True,
                                            num_workers=4,pin_memory=True,
                                            persistent_workers=True,prefetch_factor=4)
                                 for dataset in (train_ds,train_valid_ds)]
    valid_iter=DataLoader(valid_ds,batch_size,shuffle=False,drop_last=False,num_workers=4,
                          pin_memory=True,persistent_workers=True,prefetch_factor=4)
    test_iter=DataLoader(test_ds,batch_size,shuffle=False,drop_last=False,num_workers=4,
                         pin_memory=True,persistent_workers=True,prefetch_factor=4)

    trainer=Trainer(net,train_iter,train_valid_iter,valid_iter,test_iter,
                    batch_size,lr,lr_period,lr_decay,num_epochs,weight_decay,
                    pretrained=pretrained)
    trainer.train()
    trainer.plot()
    #trainer.predict(test_ds,train_valid_ds)
    #trainer.plot(val=False)

if __name__=='__main__':
    main()