# TODO: label smoothing - criterion=nn.CrossEntropyLoss(label_smoothing=0.1)
# TODO: warmup + cosine scheduler - timm.scheduler.CosineLRScheduler(warmup_t=...),
#   or torch LinearLR + CosineAnnealingLR chained with SequentialLR
import argparse
import copy
import torch
from torch import nn
from pathlib import Path
import logging

from data_loader.data_loaders import LeafDataLoader
from model.model import LeafNet
from trainer import Trainer,accuracy,top_k_acc
from utils import set_seed
from parse_config import ROOT,load_config,init_obj
from logger.logger import setup_logging
from logger.visualization import PrintCallback

def train_fold(config,fold):
    """ Train a new model with one fold of the data for validation.
    
    Args:
        config: Config dict returned by load_config().
        fold: Zero based fold number to use as the validation fold.
    
    Returns:
        Best validation accuracy for this fold.
    """
    fold_config=copy.deepcopy(config)
    fold_config['data_loader']['fold']=fold
    fold_config['trainer']['save_dir']=str(Path(config['trainer']['save_dir'])/f'fold{fold}')
    setup_logging(fold_config['trainer']['save_dir'])
    logger=logging.getLogger("train")
    logger.info("Starting fold %d",fold)

    set_seed(config['seed'])
    device_ids=list(range(torch.cuda.device_count()))
    device=torch.device('cuda',device_ids[0])

    data_loader=LeafDataLoader(**fold_config['data_loader'],seed=config['seed'])
    valid_data_loader=data_loader.split_validation()

    model=LeafNet(**config['arch']).to(device)
    if len(device_ids)>1:
        model=nn.DataParallel(model,device_ids=device_ids)
    criterion=nn.CrossEntropyLoss()
    optimizer=init_obj(config,'optimizer',torch.optim,model.parameters())
    lr_scheduler=init_obj(config,'lr_scheduler',torch.optim.lr_scheduler,optimizer,T_max=config['trainer']['epochs'])

    trainer=Trainer(model,criterion,[accuracy,top_k_acc],optimizer,fold_config,device,data_loader,valid_data_loader,lr_scheduler,callbacks=[PrintCallback()])
    trainer.train()
    return trainer.best_acc

def main(config):
    torch.backends.cudnn.benchmark=True
    model_name,n_splits=config['arch']['model_name'],config['data_loader']['n_splits']

    best_accs=[]
    for fold in range(n_splits):
        print(f"==== {model_name} fold {fold} ({fold+1}/{n_splits}) ====")
        best_accs.append(train_fold(config,fold))

    print(f"===== {model_name} {n_splits}-fold summary =====")
    for fold,acc in enumerate(best_accs):
        print(f"    fold {fold}: {acc:.4f}")
    print(f"    mean  : {sum(best_accs)/len(best_accs):.4f}")

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Train a leaf classifier')
    parser.add_argument('-c','--config',default=str(ROOT/'config.yaml'),help='path to config.yaml')
    parser.add_argument('--arch',help='timm model name, overrides arch.model_name')
    parser.add_argument('--lr',type=float,help='overrides optimizer.args.lr')
    parser.add_argument('--epochs',type=int,help='overrides trainer.epochs')
    args=parser.parse_args()
    main(load_config(args))