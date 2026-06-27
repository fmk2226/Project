import os
import pandas as pd
import torch
import torchvision
from torch import nn
import random
from pathlib import Path
from IPython import display
import DataInit

def RandomSeed(seed:int):
    random.seed(seed)
    torch.manual_seed(seed=seed)

RandomSeed(42)
display.display = lambda *args, **kwargs: None
display.clear_output = lambda *args, **kwargs: None
BASE_DIR=Path(__file__).resolve().parents[2]

#Parameters
demo=True
batch_size=32 if demo else 128
valid_ratio=0.1

data_dir=DataInit.load_data(BASE_DIR,demo=demo)
DataInit.reorg_cifar10_data(data_dir,valid_ratio)
