import math
import torch.nn as nn

from torch.nn import functional as F


class Conv3dStaticSamePadding(nn.Conv3d):
    """3D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv3dDynamicSamePadding

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 3 else [self.stride[0]] * 3

        # Calculate padding based on image size and save it
        if image_size is not None:
            id, ih, iw = (image_size, image_size, image_size) if isinstance(image_size, int) else image_size
            kd, kh, kw = self.weight.size()[-3:]
            sd, sh, sw = self.stride
            od, oh, ow = math.ceil(id / sd), math.ceil(ih / sh), math.ceil(iw / sw)

            pad_d = max((od - 1) * self.stride[0] + (kd - 1) * self.dilation[0] + 1 - id, 0)
            pad_h = max((oh - 1) * self.stride[1] + (kh - 1) * self.dilation[1] + 1 - ih, 0)
            pad_w = max((ow - 1) * self.stride[2] + (kw - 1) * self.dilation[2] + 1 - iw, 0)

            if pad_d > 0 or pad_h > 0 or pad_w > 0:
                self.static_padding = nn.ConstantPad3d((pad_w // 2, pad_w - pad_w // 2,
                                                        pad_h // 2, pad_h - pad_h // 2,
                                                        pad_d // 2, pad_d - pad_d // 2), 0)
            else:
                self.static_padding = nn.Identity()
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x
