"""预训练模型与分类头定义。"""

from __future__ import annotations

from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ViT_L_16_Weights,
    resnet18,
    resnet34,
    vit_l_16,
)


MODEL_NAMES = ("legacy_resnet34", "resnet34", "resnet18", "vit_l_16")


def _build_resnet(
    builder,
    weights,
    num_classes: int,
    dropout: float,
    legacy: bool,
) -> nn.Module:
    backbone = builder(weights=weights)
    if legacy:
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        network = nn.Sequential(
            backbone,
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        network.legacy_layout = True
        return network

    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    backbone.legacy_layout = False
    return backbone


def pretrained_resnet18(
    num_classes: int = 120, dropout: float = 0.2, legacy: bool = False
) -> nn.Module:
    return _build_resnet(
        resnet18, ResNet18_Weights.DEFAULT, num_classes, dropout, legacy
    )


def pretrained_resnet34(
    num_classes: int = 120, dropout: float = 0.2, legacy: bool = False
) -> nn.Module:
    return _build_resnet(
        resnet34, ResNet34_Weights.DEFAULT, num_classes, dropout, legacy
    )


def pretrained_vit_l_16(
    num_classes: int = 120,
    dropout: float = 0.0,
) -> nn.Module:
    """Kaggle 0.09921 高票方案所用的 ImageNet 预训练 ViT-L/16。"""
    network = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
    in_features = network.heads.head.in_features
    network.heads.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    network.legacy_layout = False
    return network


def build_model(
    model_name: str,
    num_classes: int = 120,
    dropout: float = 0.2,
) -> nn.Module:
    """按统一名称构造模型，供训练与集成脚本共同使用。"""
    if model_name == "vit_l_16":
        return pretrained_vit_l_16(num_classes, dropout)
    if model_name == "resnet18":
        return pretrained_resnet18(num_classes, dropout)
    if model_name in {"resnet34", "legacy_resnet34"}:
        return pretrained_resnet34(
            num_classes,
            dropout,
            legacy=model_name == "legacy_resnet34",
        )
    raise ValueError(
        f"unknown model {model_name!r}; expected one of {MODEL_NAMES}"
    )


def _classification_head(network: nn.Module) -> nn.Module:
    if getattr(network, "legacy_layout", False):
        return network[1:]
    if hasattr(network, "heads"):
        return network.heads
    return network.fc


def split_parameters(
    network: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """返回分类头参数和主干参数，供 Trainer 设置差分学习率。"""
    if getattr(network, "legacy_layout", False):
        backbone_parameters = list(network[0].parameters())
    else:
        backbone_parameters = []

    head_parameters = list(_classification_head(network).parameters())
    if not backbone_parameters:
        head_ids = {id(parameter) for parameter in head_parameters}
        backbone_parameters = [
            parameter
            for parameter in network.parameters()
            if id(parameter) not in head_ids
        ]
    return head_parameters, backbone_parameters
