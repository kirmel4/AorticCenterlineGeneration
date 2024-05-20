import pytest

import torch
import segmentation_models_pytorch as sm

from aaa.models.boosted_net import BoostedUnet3dSm

@pytest.mark.models
@pytest.mark.segmentation
def test_BoostedUnet3dSm_CASE_forward_AND_eval():
    IN_CHANNELS = 3
    OUT_CHANNELS = 2

    model = BoostedUnet3dSm(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        encoder_weights=None
    )

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
def test_BoostedUnet3dSm_CASE_forward_AND_train():
    IN_CHANNELS = 3
    OUT_CHANNELS = 2

    model = BoostedUnet3dSm(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        encoder_weights=None
    )

    model.train()

    x = torch.randn(1, IN_CHANNELS, 64, 64, 64)
    y, auxs = model(x)

    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == OUT_CHANNELS
    assert y.shape[2] == x.shape[2]
    assert y.shape[3] == x.shape[3]
    assert y.shape[4] == x.shape[4]

    for aux in auxs.values():
        assert aux.shape[0] == x.shape[0]
        assert aux.shape[1] == OUT_CHANNELS
        assert aux.shape[2] == x.shape[2]
        assert aux.shape[3] == x.shape[3]
        assert aux.shape[4] == x.shape[4]
