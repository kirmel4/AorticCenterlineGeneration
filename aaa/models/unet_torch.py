import torch, torch.nn as nn

class UnetTorch(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__() 

        self.unet = torch.hub.load( 'mateuszbuda/brain-segmentation-pytorch',
                                    'unet',
                                    in_channels=3,
                                    out_channels=1,
                                    init_features=32,
                                    pretrained=True )
                                    
        if in_channels != 3:
            self.unet.encoder1.enc1conv1 = nn.Conv2d( in_channels=in_channels,
                                                      out_channels=32,
                                                      kernel_size=(3, 3),
                                                      stride=(1, 1),
                                                      padding=(1, 1),
                                                      bias=False )
                                                  
        if out_channels != 1:
            self.unet.conv = nn.Conv2d( in_channels=32,
                                        out_channels=out_channels,
                                        kernel_size=(1, 1),
                                        stride=(1, 1) )

    def forward(self, inputs):
        if self.training:
            return self.unet(inputs), dict()
        else:
            return self.unet(inputs)
