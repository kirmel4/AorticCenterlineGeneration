import torch
import types
import torch.nn as nn

from collections import OrderedDict

from aaa.models.layer_convertors.misc import __classinit
from aaa.models.layer_convertors.layer_convertor import LayerConvertor


@__classinit
class LayerConvertor3d(LayerConvertor.__class__):
    @classmethod
    def _init__class(cls):
        cls._registry = {
            nn.Conv2d: getattr(cls, '_func_Conv2d'),
            nn.MaxPool2d: getattr(cls, '_func_MaxPool2d'),
            nn.AvgPool2d: getattr(cls, '_func_AvgPool2d'),
            nn.BatchNorm2d: getattr(cls, '_func_BatchNorm2d'),
            nn.AdaptiveAvgPool2d: getattr(cls, '_func_AdaptiveAvgPool2d')
        }

        return cls()

    @staticmethod
    def __expand_tuple(param):
        assert param[0] == param[1]

        return (*param, param[0])

    @classmethod
    def _func_None(cls, layer):
        return layer

    @classmethod
    def _func_AdaptiveAvgPool2d(cls, layer2d):
        kwargs = {
            'output_size': layer2d.output_size
        }

        layer3d = nn.AdaptiveAvgPool3d(**kwargs)
        return layer3d

    @classmethod
    def _func_Conv2d(cls, layer2d):
        kwargs = {
            'in_channels': layer2d.in_channels,
            'out_channels': layer2d.out_channels,
            'kernel_size': cls.__expand_tuple(layer2d.kernel_size),
            'stride': cls.__expand_tuple(layer2d.stride),
            'padding': cls.__expand_tuple(layer2d.padding),
            'dilation': cls.__expand_tuple(layer2d.dilation),
            'groups': layer2d.groups,
            'bias': 'bias' in layer2d.state_dict(),
            'padding_mode': layer2d.padding_mode
        }

        state2d = layer2d.state_dict()

        def __expand_weight(weight):
            assert weight.shape[2] == weight.shape[3]

            weight = weight.unsqueeze(dim=2)
            weight = weight.repeat((1, 1, weight.shape[-1], 1, 1))

            return weight

        state3d = OrderedDict()

        state3d['weight'] = __expand_weight(state2d['weight'])

        if 'bias' in state2d:
            state3d['bias'] = state2d['bias']

        layer3d = nn.Conv3d(**kwargs)
        layer3d.load_state_dict(state3d)

        return layer3d

    @classmethod
    def _func_BatchNorm2d(cls, layer2d):
        kwargs = {
            'num_features': layer2d.num_features,
            'eps': layer2d.eps,
            'momentum': layer2d.momentum,
            'affine': layer2d.affine,
            'track_running_stats': layer2d.track_running_stats
        }

        layer3d = nn.BatchNorm3d(**kwargs)

        return layer3d

    @classmethod
    def _func_MaxPool2d(cls, layer2d):
        kwargs = {
            'kernel_size': layer2d.kernel_size,
            'stride': layer2d.stride,
            'padding': layer2d.padding,
            'dilation': layer2d.dilation,
            'return_indices': layer2d.return_indices,
            'ceil_mode': layer2d.ceil_mode
        }

        layer3d = nn.MaxPool3d(**kwargs)

        return layer3d

    @classmethod
    def _func_AvgPool2d(cls, layer2d):
        kwargs = {
            'kernel_size': layer2d.kernel_size,
            'stride': layer2d.stride,
            'padding': layer2d.padding,
            'ceil_mode': layer2d.ceil_mode,
            'count_include_pad': layer2d.count_include_pad,
            'divisor_override': layer2d.divisor_override
        }

        layer3d = nn.AvgPool3d(**kwargs)

        return layer3d
