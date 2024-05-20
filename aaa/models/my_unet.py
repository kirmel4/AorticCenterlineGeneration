import torch, torch.nn as nn
import segmentation_models_pytorch as sm

from aaa.models.layer_convertors import convert_inplace, LayerConvertorSm, LayerConvertor3d, LayerConvertorPoolCpu

class my_unet(sm.Unet):
    def __init__(self, in_channels = 1, **kwargs):
        super().__init__(in_channels=in_channels, **kwargs)

        convert_inplace(self, LayerConvertorSm)
        convert_inplace(self, LayerConvertor3d)
        convert_inplace(self, LayerConvertorPoolCpu)
        
        self.segmentation_head = None


        self.head1 = nn.Sequential(torch.nn.Conv3d(16, 8, kernel_size=(3,3,3), padding=(1,1,1)),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   torch.nn.Conv3d(8, 4, kernel_size=(3,3,3), padding=(1,1,1)),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   torch.nn.Conv3d(4, 2, kernel_size=(3,3,3), padding=(1,1,1)))
        
        self.head2 = nn.Sequential(torch.nn.Conv3d(16, 8, kernel_size=(3,3,3), padding=(1,1,1)),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   torch.nn.Conv3d(8, 4, kernel_size=(3,3,3), padding=(1,1,1)),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   torch.nn.Conv3d(4, 3, kernel_size=(3,3,3), padding=(1,1,1)))


    def forward(self, inputs):
        features = self.encoder(inputs)#torch.unsqueeze(inputs, 0))

        decoder_output = self.decoder(*features)
        
        output1 = self.head1(decoder_output)
        output2 = self.head2(decoder_output)



        return output1, output2