import torch.nn as nn
from functools import partial
from timm.models.efficientnet import round_channels, default_cfgs, decode_arch_def

from segmentation_models_pytorch.encoders.timm_efficientnet import EfficientNetBaseEncoder, EfficientNetEncoder

from aaa.models.encoders.misc import prepare_settings

def get_efficientnetv2_m_kwargs(channel_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2):
    ''' Creates an EfficientNet-V2 Medium model

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298
    '''

    arch_def = [
        ['cn_r3_k3_s1_e1_c24_skip'],
        ['er_r5_k3_s2_e4_c48'],
        ['er_r5_k3_s2_e4_c80'],
        ['ir_r7_k3_s2_e4_c160_se0.25'],
        ['ir_r14_k3_s1_e6_c176_se0.25'],
        ['ir_r18_k3_s2_e6_c304_se0.25'],
        ['ir_r5_k3_s1_e6_c512_se0.25'],
    ]

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=1280,
        stem_size=24,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        act_layer=nn.SiLU,
        drop_rate=drop_rate,
        drop_path_rate=0.2
    )

    return model_kwargs

def get_efficientnetv2_l_kwargs(channel_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2):
    ''' Creates an EfficientNet-V2 Large model

    Ref impl: https://github.com/google/automl/tree/master/efficientnetv2
    Paper: `EfficientNetV2: Smaller Models and Faster Training` - https://arxiv.org/abs/2104.00298
    '''

    arch_def = [
        ['cn_r4_k3_s1_e1_c32_skip'],
        ['er_r7_k3_s2_e4_c64'],
        ['er_r7_k3_s2_e4_c96'],
        ['ir_r10_k3_s2_e4_c192_se0.25'],
        ['ir_r19_k3_s1_e6_c224_se0.25'],
        ['ir_r25_k3_s2_e6_c384_se0.25'],
        ['ir_r7_k3_s1_e6_c640_se0.25'],
    ]

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=1280,
        stem_size=32,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        act_layer=nn.SiLU,
        drop_rate=drop_rate,
        drop_path_rate=0.2
    )

    return model_kwargs

class EfficientNetV2MEncoder(EfficientNetBaseEncoder):
    def __init__(
        self,
        stage_idxs,
        out_channels,
        depth=5,
        channel_multiplier=1.0,
        depth_multiplier=1.0,
        drop_rate=0.2,
    ):
        kwargs = get_efficientnetv2_m_kwargs(channel_multiplier, depth_multiplier, drop_rate)
        super().__init__(stage_idxs, out_channels, depth, **kwargs)

class EfficientNetV2LEncoder(EfficientNetBaseEncoder):
    def __init__(
        self,
        stage_idxs,
        out_channels,
        depth=5,
        channel_multiplier=1.0,
        depth_multiplier=1.0,
        drop_rate=0.2,
    ):
        kwargs = get_efficientnetv2_l_kwargs(channel_multiplier, depth_multiplier, drop_rate)
        super().__init__(stage_idxs, out_channels, depth, **kwargs)

timm_efficientnet_encoders = {
    'timm-tf_efficientnetv2_m': {
        'encoder': EfficientNetV2MEncoder,
        'pretrained_settings': {
            'imagenet': prepare_settings(default_cfgs['tf_efficientnetv2_m'].cfgs['in1k']),
        },
        'params': {
            'out_channels': (3, 24, 48, 80, 176, 512),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.0,
            'depth_multiplier': 1.0,
            'drop_rate': 0.2,
        },
    },
    'timm-tf_efficientnetv2_l': {
        'encoder': EfficientNetV2LEncoder,
        'pretrained_settings': {
            'imagenet': prepare_settings(default_cfgs['tf_efficientnetv2_l'].cfgs['in1k']),
        },
        'params': {
            'out_channels': (3, 32, 64, 96, 224, 640),
            'stage_idxs': (2, 3, 5),
            'channel_multiplier': 1.0,
            'depth_multiplier': 1.0,
            'drop_rate': 0.2,
        },
    },
}
