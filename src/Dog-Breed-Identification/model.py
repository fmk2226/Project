from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.models import ResNet18_Weights, ResNet34_Weights

def pretrained_resnet18():
    finetune_net=nn.Sequential()
    finetune_net.features=torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    finetune_net.output_new=nn.Sequential(nn.Linear(1000,256),nn.ReLU(),nn.Linear(256,120))
    for param in finetune_net.features.parameters():
        param.requires_grad=False
    return finetune_net

def pretrained_resnet34():
    finetune_net=nn.Sequential()
    finetune_net.features=torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
    finetune_net.output_new=nn.Sequential(nn.Linear(1000,256),nn.ReLU(),nn.Linear(256,120))
    for param in finetune_net.features.parameters():
        param.requires_grad=False
    return finetune_net