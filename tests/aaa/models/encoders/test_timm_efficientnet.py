import pytest

import torch

from aaa.models.unet3d_sm import Unet3dSm, Unetpp3dSm

@pytest.mark.models
@pytest.mark.segmentation
def test_EfficientNetV2MEncoder_CASE_creation_Unet3dSm():
    IN_CHANNELS = 3
    OUT_CHANNELS = 2

    model = Unet3dSm( in_channels=IN_CHANNELS,
                      out_channels=OUT_CHANNELS,
                      encoder_name='timm-tf_efficientnetv2_m' )
    model.eval()

    x = torch.randn(1, IN_CHANNELS, 64, 64, 64)
    y = model(x)

    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == OUT_CHANNELS
    assert y.shape[2] == x.shape[2]
    assert y.shape[3] == x.shape[3]
    assert y.shape[4] == x.shape[4]

@pytest.mark.models
@pytest.mark.segmentation
def test_EfficientNetV2MEncoder_CASE_init_Unet3dSm():
    IN_CHANNELS = 3
    OUT_CHANNELS = 2

    model = Unet3dSm( in_channels=IN_CHANNELS,
                      out_channels=OUT_CHANNELS,
                      encoder_name='timm-tf_efficientnetv2_m',
                      encoder_weights='imagenet' )
    model.eval()

    x = torch.randn(1, IN_CHANNELS, 64, 64, 64)
    y = model(x)

    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == OUT_CHANNELS
    assert y.shape[2] == x.shape[2]
    assert y.shape[3] == x.shape[3]
    assert y.shape[4] == x.shape[4]

@pytest.mark.models
@pytest.mark.segmentation
def test_EfficientNetV2MEncoder_CASE_creation_Unet3dSm():
    IN_CHANNELS = 3
    OUT_CHANNELS = 2

    model = Unet3dSm( in_channels=IN_CHANNELS,
                      out_channels=OUT_CHANNELS,
                      encoder_name='timm-tf_efficientnetv2_l' )
    model.eval()

    x = torch.randn(1, IN_CHANNELS, 64, 64, 64)
    y = model(x)

    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == OUT_CHANNELS
    assert y.shape[2] == x.shape[2]
    assert y.shape[3] == x.shape[3]
    assert y.shape[4] == x.shape[4]

@pytest.mark.models
@pytest.mark.segmentation
def test_EfficientNetV2MEncoder_CASE_creation_Unetpp3dSm():
    IN_CHANNELS = 3
    OUT_CHANNELS = 2

    model = Unetpp3dSm( in_channels=IN_CHANNELS,
                        out_channels=OUT_CHANNELS,
                        encoder_name='timm-tf_efficientnetv2_m' )
    model.eval()

    x = torch.randn(1, IN_CHANNELS, 64, 64, 64)
    y = model(x)

    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == OUT_CHANNELS
    assert y.shape[2] == x.shape[2]
    assert y.shape[3] == x.shape[3]
    assert y.shape[4] == x.shape[4]

@pytest.mark.models
@pytest.mark.segmentation
def test_EfficientNetV2MEncoder_CASE_creation_Unetpp3dSm():
    IN_CHANNELS = 3
    OUT_CHANNELS = 2

    model = Unetpp3dSm( in_channels=IN_CHANNELS,
                        out_channels=OUT_CHANNELS,
                        encoder_name='timm-tf_efficientnetv2_l' )
    model.eval()

    x = torch.randn(1, IN_CHANNELS, 64, 64, 64)
    y = model(x)

    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == OUT_CHANNELS
    assert y.shape[2] == x.shape[2]
    assert y.shape[3] == x.shape[3]
    assert y.shape[4] == x.shape[4]
