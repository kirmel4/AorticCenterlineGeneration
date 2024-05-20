import torch
import types
import torch.nn as nn

from collections import OrderedDict

from aaa.models.layer_convertors.misc import __classinit
from aaa.models.layer_convertors.layer_convertor import LayerConvertor

@__classinit
class LayerConvertorPoolCpu(LayerConvertor.__class__):
    @classmethod
    def _init__class(cls):
        cls._registry = {
            nn.MaxPool3d: getattr(cls, '_func_MaxPool3d'),
        }

        return cls()

    @classmethod
    def _func_MaxPool3d(cls, layer2d):
        kwargs = {
            'kernel_size': layer2d.kernel_size,
            'stride': layer2d.stride,
            'padding': layer2d.padding,
            'dilation': layer2d.dilation,
            'return_indices': layer2d.return_indices,
            'ceil_mode': layer2d.ceil_mode
        }

        class MaxPool3dCpu(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()

                self.layer = nn.MaxPool3d(**kwargs)

            def forward(self, x):
                return self.layer.cpu()(x.cpu()).to(x.device)

        layer3d = MaxPool3dCpu(**kwargs)

        return layer3d
