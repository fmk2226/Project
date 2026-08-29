from torch import nn
from torch.nn import functional as F
import torchvision

def pretrained_resnet18(devices):
    finetune_net=nn.Sequential()
    finetune_net.features=torchvision.models.resnet18(pretrained=True)
    finetune_net.output_new=nn.Sequential(nn.Linear(1000,256),nn.ReLU(),nn.Linear(256,120))
    finetune_net=finetune_net.to(devices[0])
    for param in finetune_net.features.parameters():
        param.requires_grad=False
    return finetune_net

def pretrained_resnet34(devices):
    finetune_net=nn.Sequential()
    finetune_net.features=torchvision.models.resnet34(pretrained=True)
    finetune_net.output_new=nn.Sequential(nn.Linear(1000,256),nn.ReLU(),nn.Linear(256,120))
    finetune_net=finetune_net.to(devices[0])
    for param in finetune_net.features.parameters():
        param.requires_grad=False
    return finetune_net