from torch import nn
import timm

class LeafNet(nn.Module):
    """ Leaf classifier built from timm pretrained model with optional MLP classifier

    Args:
        model_name: Name of model in timm package.
        num_classes: Number of classes to classify.
        pretrained: Boolean indicates whether to use pretrained net or not.
        use_mlp: Boolean indicates whether to use MLP as classifier or not.

    Attributes:
        Pretrained timm model with optional MLP classifier
    """
    def __init__(self,model_name,num_classes=176,pretrained=True,use_mlp=False,**model_kwargs):
        super().__init__()
        self.use_mlp=use_mlp
        self.model=timm.create_model(model_name,pretrained=pretrained,num_classes=0 if use_mlp else num_classes,**model_kwargs)
        if use_mlp:
            n_features=self.model.num_features
            self.classifier=nn.Sequential(nn.Linear(n_features,4096),nn.ReLU(),nn.Dropout(0.3),
                                          nn.Linear(4096,2048),nn.ReLU(),nn.Dropout(0.1),
                                          nn.Linear(2048,num_classes)
                                          )

    def forward(self,x):
        """Return class logits with shape [batch_size,num_classes]"""
        x=self.model(x)
        if self.use_mlp:
            x=self.classifier(x)
        return x