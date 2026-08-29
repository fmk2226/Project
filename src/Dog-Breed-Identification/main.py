import os
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import random
from pathlib import Path
from IPython import display
import Data_Prep
from model import pretrained_resnet18,pretrained_resnet34
from Trainer import Trainer

def RandomSeed(seed:int):
    os.environ["PYTHONHASHSEED"]=str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)

def seed_worker(worker_id):
    worker_seed=torch.initial_seed()%2**32 #map torch's seed to np's range
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    display.display = lambda *args, **kwargs: None
    display.clear_output = lambda *args, **kwargs: None
    BASE_DIR=Path(__file__).resolve().parents[2]

    #parameters
    demo=True
    use_resnet34=True
    seed=42
    num_workers=4
    valid_ratio=0.1
    batch_size=32 if demo else 256
    lr,lr_period,lr_decay=1e-3,2,0.9
    num_epochs=10
    weight_decay=1e-4
    net=pretrained_resnet34 if use_resnet34 else pretrained_resnet18

    #seed everything
    RandomSeed(seed)
    generator=torch.Generator().manual_seed(seed)

    #initialize dataset to a readable form for torchvision
    data_dir=Data_Prep.load_data(BASE_DIR,demo=demo)
    Data_Prep.reorg_dog_breed_data(data_dir,valid_ratio)

    #data augmentation
    normalize=torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    transform_train=torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224,scale=(0.08,1),ratio=(3/4,4/3)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
        torchvision.transforms.ToTensor(),
        normalize
    ])
    transfrom_test=torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize
    ])

    train_ds,train_valid_ds=[
        torchvision.datasets.ImageFolder(os.path.join(data_dir,'train_valid_test',folder),transform=transform_train)
        for folder in ['train','train_valid']
        ]
    valid_ds,test_ds=[
        torchvision.datasets.ImageFolder(os.path.join(data_dir,'train_valid_test',folder),transform=transfrom_test)
        for folder in ['valid','test']
    ]

    #define data iterator
    train_iter,train_valid_iter=[
        DataLoader(dataset,batch_size,shuffle=True,drop_last=True,num_workers=num_workers,
                   pin_memory=True,worker_init_fn=seed_worker,generator=generator)
        for dataset in (train_ds,train_valid_ds)
    ]
    valid_iter=DataLoader(valid_ds,batch_size,shuffle=False,drop_last=False,num_workers=num_workers,
                          pin_memory=True,worker_init_fn=seed_worker,generator=generator)
    test_iter=DataLoader(test_ds,batch_size,shuffle=False,drop_last=False,num_workers=num_workers,
                         pin_memory=True,worker_init_fn=seed_worker,generator=generator)

    trainer=Trainer(pretrained_resnet34,train_iter,train_valid_iter,valid_iter,test_iter,
                    batch_size,lr,lr_period,lr_decay,num_epochs,weight_decay)
    trainer.train()
    trainer.plot()

if __name__=='__main__':
    main()