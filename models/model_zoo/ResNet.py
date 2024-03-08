from torch import nn
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights


def get_resnet():
    pretrained_net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    finetune_net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    finetune_net.fc = nn.Linear(pretrained_net.fc.in_features, 176)
    return finetune_net
