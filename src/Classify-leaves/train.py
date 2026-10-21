# TODO: label smoothing - criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
# TODO: warmup + cosine scheduler - timm.scheduler.CosineLRScheduler(warmup_t=...),
#   or torch LinearLR + CosineAnnealingLR chained with SequentialLR
import torch
from torch import nn
from pathlib import Path

from data_loader.data_loaders import LeafDataLoader
from model.model import LeafNet
from trainer import Trainer,accuracy,top_k_acc
from utils import set_seed

ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT.parents[1]/'data'/'classify_leaves'

MODEL_NAME='resnet50d'
FOLD=0

config={
    'seed':42,
    'arch':{'model_name':'resnet50d','num_classes':176,'pretrained':True},
    'data_loader':{'csv_file':str(DATA_DIR/'train.csv'),'data_dir':str(DATA_DIR),
                   'batch_size':64,'fold':0,'n_splits':5,'num_workers':8},
    'optimizer':{'lr':1e-3,'weight_decay':1e-5},
    'trainer':{'epochs':30,'save_dir':str(ROOT/'saved'/'model_weights'/MODEL_NAME/f'fold{FOLD}')}
}

def main(config):
    set_seed(config['seed'])
    torch.backends.cudnn.benchmark=True
    device=torch.device('cuda')

    data_loader=LeafDataLoader(**config['data_loader'],seed=config['seed'])
    valid_data_loader=data_loader.split_validation()

    model=LeafNet(**config['arch']).to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.AdamW(model.parameters(),**config['optimizer'])
    lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['trainer']['epochs'])

    trainer=Trainer(model,criterion,[accuracy,top_k_acc],optimizer,config,device,data_loader,valid_data_loader,lr_scheduler)
    trainer.train()

if __name__=='__main__':
    main(config)