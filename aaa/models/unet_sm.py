import torch, torch.nn as nn
import segmentation_models_pytorch as sm

from aaa.models.layer_convertors import convert_inplace, LayerConvertorSm

class UnetSm(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super().__init__()

        self.unet = sm.Unet(in_channels=in_channels, classes=out_channels, **kwargs)
        convert_inplace(self.unet, LayerConvertorSm)

    def forward(self, inputs):
        if self.training:
            return self.unet(inputs), dict()
        else:
            return self.unet(inputs)

class UnetppSm(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super().__init__()

        self.unetpp = sm.UnetPlusPlus(in_channels=in_channels, classes=out_channels, **kwargs)
        convert_inplace(self.unetpp, LayerConvertorSm)

    def forward(self, inputs):
        if self.training:
            return self.unetpp(inputs), dict()
        else:
            return self.unetpp(inputs)
