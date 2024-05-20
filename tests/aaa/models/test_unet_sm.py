import pytest

import torch
import segmentation_models_pytorch as sm

from aaa.models.unet_sm import UnetSm, UnetppSm

@pytest.mark.models
@pytest.mark.segmentation
def test_UnetSm_CASE_reusage_with_sm():
    IN_CHANNELS = 3
    OUT_CHANNELS = 2
    ATTENTION = 'scse'

    model = UnetSm(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        decoder_attention_type=ATTENTION,
        encoder_weights=None
    )

    libmodel = sm.Unet(
        in_channels=IN_CHANNELS,
        classes=OUT_CHANNELS,
        decoder_attention_type=ATTENTION,
        encoder_weights=None
    )

    state_dict = model.state_dict()
    state_dict = { key[5:]: state_dict[key] for key in model.state_dict() }

    libmodel.load_state_dict(state_dict)

@pytest.mark.models
@pytest.mark.segmentation
def test_UnetSm_CASE_forward_AND_eval():
    IN_CHANNELS = 3
    OUT_CHANNELS = 2

    model = UnetSm(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        encoder_weights=None
    )

    model.eval()

    x = torch.randn(1, IN_CHANNELS, 64, 64)
    y = model(x)

    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == OUT_CHANNELS
    assert y.shape[2] == x.shape[2]
    assert y.shape[3] == x.shape[3]

@pytest.mark.models
@pytest.mark.segmentation
def test_UnetppSm_CASE_forward_AND_eval():
    IN_CHANNELS = 3
    OUT_CHANNELS = 2

    model = UnetppSm(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        encoder_weights=None
    )

    model.eval()

    x = torch.randn(1, IN_CHANNELS, 64, 64)
    y = model(x)

    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == OUT_CHANNELS
    assert y.shape[2] == x.shape[2]
    assert y.shape[3] == x.shape[3]
