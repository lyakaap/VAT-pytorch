import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


def _pool_flatten(x, how_pool='avg'):
    if how_pool == 'avg':
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
    elif how_pool == 'max':
        x = F.adaptive_max_pool2d(x, 1).squeeze()
    else:
        raise ValueError
    return x


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, how_pool='avg', dropout=0.0, pretrained=False):
        super(ResNet18, self).__init__()

        self.num_classes = num_classes
        self.base_model = nn.Sequential(*list(
            resnet18(pretrained=pretrained).children())[:-2])
        self.linear = nn.Linear(512, num_classes)
        self.how_pool = how_pool
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x):
        x = self.base_model(x)
        x = _pool_flatten(x, how_pool=self.how_pool)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.linear(x)

        return x
