import torch, torch.nn as nn
import segmentation_models_pytorch as sm

from aaa.models.layer_convertors import convert_inplace, LayerConvertorSm, LayerConvertor3d, LayerConvertorPoolCpu

class Unet3dSm(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super().__init__()

        self.unet = sm.Unet(in_channels=in_channels, classes=out_channels, **kwargs)
        self.unet.check_input_shape = lambda *args: None

        convert_inplace(self.unet, LayerConvertorSm)
        convert_inplace(self.unet, LayerConvertor3d)
        convert_inplace(self.unet, LayerConvertorPoolCpu)

    def forward(self, inputs):
        if self.training:
            return self.unet(inputs), dict()
        else:
            return self.unet(inputs)

class Unetpp3dSm(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super().__init__()

        self.unetpp = sm.UnetPlusPlus(in_channels=in_channels, classes=out_channels, **kwargs)
        convert_inplace(self.unetpp, LayerConvertorSm)
        convert_inplace(self.unetpp, LayerConvertor3d)
        convert_inplace(self.unetpp, LayerConvertorPoolCpu)

    def forward(self, inputs):
        if self.training:
            return self.unetpp(inputs), dict()
        else:
            return self.unetpp(inputs)
