import torch
import types
import numpy as np
import torch.nn as nn
import efficientnet_pytorch as ep
import segmentation_models_pytorch as sm

from collections import OrderedDict
from torch.nn import functional as F

from aaa.models.additional_layers import Conv3dStaticSamePadding

def __classinit(cls):
    return cls._init__class()

@__classinit
class LayerConvertor(object):
    @classmethod
    def _init__class(cls):
        cls._registry = {
            sm.unet.decoder.DecoderBlock: getattr(cls, '_func_sm_unet_DecoderBlock'),
            sm.unetplusplus.decoder.DecoderBlock: getattr(cls, '_func_sm_unetplusplus_DecoderBlock'),
            ep.utils.Conv2dStaticSamePadding: getattr(cls, '_func_ep_Conv2dStaticSamePadding'),
            ep.model.MBConvBlock: getattr(cls, '_func_ep_MBConvBlock')
        }

        return cls()

    def __call__(self, layer):
        if type(layer) in self._registry:
            return self._registry[type(layer)](layer)
        else:
            return self._func_None(layer)

    @staticmethod
    def __expand_tuple(param):
        assert param[0] == param[1]

        return (*param, param[0])

    @classmethod
    def _func_None(cls, layer):
        return layer

    @staticmethod
    def _sm_unet_decoder_forward(self, x, skip=None):
        if skip is not None:
            scale_factor = list()

            getdim = lambda vector, axis : vector.shape[axis]

            naxis = len(x.shape)
            for axis in np.arange(2, naxis):
                scale_factor.append(getdim(skip, axis)/getdim(x, axis))

            scale_factor = tuple(scale_factor)
        else:
            scale_factor = 2

        x = nn.functional.interpolate(x, scale_factor=scale_factor, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

    @classmethod
    def _func_sm_unet_DecoderBlock(cls, layer):
        layer.forward = types.MethodType(cls._sm_unet_decoder_forward, layer)

        return layer

    @classmethod
    def _func_sm_unetplusplus_DecoderBlock(cls, layer):
        layer.forward = types.MethodType(cls._sm_unet_decoder_forward, layer)

        return layer

    @classmethod
    def _func_ep_Conv2dStaticSamePadding(cls, layer2d):
        kwargs = {
            'in_channels': layer2d.in_channels,
            'out_channels': layer2d.out_channels,
            'kernel_size': cls.__expand_tuple(layer2d.kernel_size),
            'stride': cls.__expand_tuple(layer2d.stride),
            'padding': cls.__expand_tuple(layer2d.padding),
            'dilation': cls.__expand_tuple(layer2d.dilation),
            'groups': layer2d.groups,
            'bias': 'bias' in layer2d.state_dict(),
            'padding_mode': layer2d.padding_mode,
            'image_size': None
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

        layer3d = Conv3dStaticSamePadding(**kwargs)
        layer3d.load_state_dict(state3d)

        if isinstance(layer2d.static_padding, nn.ZeroPad2d):
            assert layer2d.static_padding.padding[0] == layer2d.static_padding.padding[2]
            assert layer2d.static_padding.padding[1] == layer2d.static_padding.padding[3]

            layer3d.static_padding = nn.ConstantPad3d(( *layer2d.static_padding.padding,
                                                         layer2d.static_padding.padding[0],
                                                         layer2d.static_padding.padding[1] ), 0)

        return layer3d

    @staticmethod
    def _ep_MBConvBlock_forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool3d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = ep.utils.drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    @classmethod
    def _func_ep_MBConvBlock(cls, layer):
        layer.forward = types.MethodType(cls._ep_MBConvBlock_forward, layer)

        return layer
