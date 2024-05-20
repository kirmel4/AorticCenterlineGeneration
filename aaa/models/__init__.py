from segmentation_models_pytorch.encoders import encoders as encodersSm

from aaa.models.unet_torch import UnetTorch
from aaa.models.unet_sm import UnetSm, UnetppSm

from aaa.models.unet3d_sm import Unet3dSm, Unetpp3dSm
from aaa.models.unet3d_monai import UnetTr3dMonai, SwinUnet3dMonai, Vnet3dMonai

from aaa.models.boosted_net import BoostedUnetSm, BoostedUnetppSm, BoostedUnet3dSm, BoostedUnetpp3dSm

from aaa.models.encoders.timm_efficientnet import timm_efficientnet_encoders

encodersSm.update(timm_efficientnet_encoders)

