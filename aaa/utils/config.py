import sys
import time
import torch

from tqdm import tqdm
from pathlib import Path

from aaa.utils import load_yaml_config
from aaa.utils.tqdm import TqdmLogger

def init_kwargs(config, kwargs):
    for key, value in kwargs.items():
        if key.upper().endswith('PATH'):
            if value is not None:
                config[key.upper()] = Path(value)
            else:
                raise RuntimeError(f'Path option {key} is None')
        else:
            config[key.upper()] = value

def init_device(config):
    if torch.cuda.is_available():
        config['DEVICE'] = torch.device('cuda')
    else:
        config['DEVICE'] = torch.device('cpu')

def init_verboser(config, **kwargs):
    if config['VERBOSE']:
        file_ = TqdmLogger(kwargs.get('logger')) if 'logger' in kwargs else None
        config['VERBOSER'] = lambda x, **lkwargs: tqdm(x, file=file_, **lkwargs)
    else:
        config['VERBOSER'] = lambda x, **lkwargs: x

def init_options(config):
    for key in list(config.keys()):
        if key.endswith('OPTIONS_PATH'):
            option_key = key[:-5]
            config[option_key] = load_yaml_config(config[key])

def init_run_command(config):
    config['SCRIPT'] = ' '.join(sys.argv)

def init_timestamp(config):
    config['PREFIX'] = time.strftime("%d-%m-%y:%H-%M_", time.gmtime())
