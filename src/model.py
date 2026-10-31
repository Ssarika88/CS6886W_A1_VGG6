# model.py
# VGG6 model for CIFAR-10 with the activation function made configurable so we can try different ones.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(name):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'silu':
        return nn.SiLU(inplace=True)
    elif name == 'gelu':
        return nn.GELU()
    else:
        raise ValueError("Unknown activation: " + name)

class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def make_layers(cfg, activation='relu', batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), get_activation(activation)]
            else:
                layers += [conv2d, get_activation(activation)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg(cfg, num_classes=10, activation='relu', batch_norm=True):
    return VGG(make_layers(cfg, activation=activation, batch_norm=batch_norm),
               num_classes=num_classes)

cfg_vgg6 = [64, 64, 'M', 128, 128, 'M']
