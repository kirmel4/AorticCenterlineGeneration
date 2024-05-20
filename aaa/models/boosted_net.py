import torch, torch.nn as nn

from aaa.models.unet_sm import UnetSm, UnetppSm
from aaa.models.unet3d_sm import Unet3dSm, Unetpp3dSm

def BoostedNet(Type):
    class BoostedNet(nn.Module):
        def __init__(self, ntree=3, in_channels=3, out_channels=1, **kwargs):
            super().__init__()

            self.__suffix = 'net_{}'
            self.__ntree = ntree

            self.nets = nn.ModuleDict()

            self.nets[self.__suffix.format(0)] = Type(in_channels=in_channels, out_channels=out_channels, **kwargs)

            for idx in range(1, self.__ntree):
                name = self.__suffix.format(len(self.nets))
                self.nets[name] = Type(in_channels=out_channels, out_channels=out_channels, **kwargs)

        def forward(self, inputs):
            output = (inputs, )

            auxs = {}

            for idx in range(self.__ntree):
                name = self.__suffix.format(idx)

                if self.training:
                    output = (*output, self.nets[name](output[-1])[0])
                else:
                    output = (*output, self.nets[name](output[-1]))

                auxs[self.__suffix.format(idx)] = output[-1]

            masks = output[-1]
            del auxs[self.__suffix.format(self.__ntree-1)]

            if self.training:
                return masks, auxs
            else:
                return masks

    return BoostedNet

BoostedUnetSm = BoostedNet(UnetSm)
BoostedUnetppSm = BoostedNet(UnetppSm)

BoostedUnet3dSm = BoostedNet(Unet3dSm)
BoostedUnetpp3dSm = BoostedNet(Unetpp3dSm)
