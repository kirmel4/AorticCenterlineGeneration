import timm
import torch
import types
import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as sm

from aaa.models.layer_convertors.misc import __classinit
from aaa.models.layer_convertors.layer_convertor import LayerConvertor


@__classinit
class LayerConvertorSm(LayerConvertor.__class__):
    @classmethod
    def _init__class(cls):
        cls._registry = {
            sm.decoders.unet.decoder.DecoderBlock: getattr(cls, '_func_sm_unet_decoder_DecoderBlock'),
            sm.decoders.unet.decoder.UnetDecoder: getattr(cls, '_func_sm_unet_decoder_UnetDecoder'),
            sm.decoders.unetplusplus.decoder.DecoderBlock: getattr(cls, '_func_sm_unetplusplus_decoder_DecoderBlock'),
            timm.layers.norm_act.BatchNormAct2d: getattr(cls, '_func_timm_layers_norm_act_BatchNormAct2d')
        }

        return cls()

    @staticmethod
    def _sm_unet_decoder_decoderblock_forward(self, x, skip=None, shape=None):
        if shape is not None:
            scale_factor = list()

            getdim = lambda vector, axis : vector.shape[axis]

            naxis = len(x.shape)
            for axis in np.arange(2, naxis):
                scale_factor.append(shape[axis]/getdim(x, axis))

            scale_factor = tuple(scale_factor)
        else:
            scale_factor = 2

        x = nn.functional.interpolate(x, scale_factor=scale_factor, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)

        return x

    @staticmethod
    def _sm_unet_decoder_unetdecoder_forward(self, *features):
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            if i < len(skips) - 1:
                skip = skips[i]
                shape = skips[i].shape
            else:
                skip = None
                shape = skips[i].shape

            x = decoder_block(x, skip, shape)

        return x

    @staticmethod
    def _timm_layers_norm_act_batchnormact2d_forward(self, x):
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        x = nn.functional.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

        x = self.drop(x)
        x = self.act(x)

        return x

    @classmethod
    def _func_sm_unet_decoder_DecoderBlock(cls, layer):
        layer.forward = types.MethodType(cls._sm_unet_decoder_decoderblock_forward, layer)

        return layer

    @classmethod
    def _func_sm_unet_decoder_UnetDecoder(cls, layer):
        layer.forward = types.MethodType(cls._sm_unet_decoder_unetdecoder_forward, layer)

        return layer

    @classmethod
    def _func_sm_unetplusplus_decoder_DecoderBlock(cls, layer):
        layer.forward = types.MethodType(cls._sm_unet_decoder_decoderblock_forward, layer)

        return layer

    @classmethod
    def _func_timm_layers_norm_act_BatchNormAct2d(cls, layer):
        layer.forward = types.MethodType(cls._timm_layers_norm_act_batchnormact2d_forward, layer)

        return layer
