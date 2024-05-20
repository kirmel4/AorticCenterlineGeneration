
import torch, torch.nn as nn
from monai.networks.nets import UNETR, SwinUNETR, VNet

class UnetTr3dMonai(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, voxel_shape=(64, 64, 64), **kwargs):
        super().__init__()

        self.unet = UNETR(in_channels=in_channels, out_channels=out_channels, img_size=voxel_shape, **kwargs)

    def forward(self, inputs):
        if self.training:
            return self.unet(inputs), dict()
        else:
            return self.unet(inputs)

class SwinUnet3dMonai(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, voxel_shape=(64, 64, 64), **kwargs):
        super().__init__()

        self.unet = SwinUNETR(in_channels=in_channels, out_channels=out_channels, img_size=voxel_shape, **kwargs)

    def forward(self, inputs):
        if self.training:
            return self.unet(inputs), dict()
        else:
            return self.unet(inputs)

class Vnet3dMonai(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, **kwargs):
        super().__init__()

        self.vnet = VNet( in_channels=in_channels,
                          out_channels=out_channels,
                          spatial_dims=3,
                          **kwargs )

    def forward(self, inputs):
        if self.training:
            return self.unet(inputs), dict()
        else:
            return self.unet(inputs)
