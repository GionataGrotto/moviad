"""
PathcDescriptor module used with CFA
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from moviad.models.audio.components.cfa.coordconv import CoordConv2d

SUPPORTED_BACKBONES = ("vgg19_bn", "resnet18", "wide_resnet50_2", "efficientnet_b5")

class Descriptor(torch.nn.Module):
    def __init__(self, gamma_d, feature_map_channels, cnn):
        super(Descriptor, self).__init__()
        self.cnn = cnn
        dim = feature_map_channels
        self.layer = CoordConv2d(dim, dim//gamma_d, 1)


    def forward(self, p):
        sample = None
        if isinstance(p, list):
            for o in p:
                o = F.avg_pool2d(o, 3, 1, 1) / o.size(1) if self.cnn == 'efficientnet_b5' else F.avg_pool2d(o, 3, 1, 1)
                sample = o if sample is None else torch.cat((sample, F.interpolate(o, (sample.size(2),sample.size(3)), mode='bilinear')), dim=1)
        else:
            sample = p

        phi_p = self.layer(sample)
        return phi_p