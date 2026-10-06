from torch import nn
import timm

class LeafNet(nn.Module):
    """ Leaf classifier built from timm pretrained model with MLP classifier

    Args:
        model_name: Name of model in timm package.
        num_classes: Number of classes to classify.
        pretrained: Boolean indicates whether to use pretrained net or not.

    Attributes:
        Pretrained timm model using MLP as classifier
    """
    def __init__(self,model_name,num_classes=176,pretrained=True):
        super().__init__()
        self.model=timm.create_model(model_name,pretrained=pretrained)
        n_features=self.model.fc.in_features
        fc=nn.Sequential(
            nn.Linear(n_features,2048),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(2048,2048),nn.ReLU(),nn.Dropout(0.1),
            nn.Linear(2048,num_classes)
        )
        self.model.fc=fc

    def forward(self,x):
        """Return class logits with shape [batch_size,num_classes]"""
        return self.model(x)