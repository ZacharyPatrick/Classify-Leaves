from torch import nn
from .model_zoo.AlexNet import AlexNet
from .model_zoo.VGG import VGG
from .model_zoo.ResNet import get_resnet


class Model(nn.Module):

    def __init__(self, num_classes, model='AlexNet', conv_arch=None):
        super(Model, self).__init__()
        if conv_arch is None:
            conv_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
        if model == 'AlexNet':
            self.net = AlexNet(num_classes)
        elif model == 'VGG':
            self.net = VGG(conv_arch=conv_arch, num_classes=num_classes)
        elif model == 'ResNet':
            self.net = get_resnet()

    def forward(self, x):
        return self.net(x)
