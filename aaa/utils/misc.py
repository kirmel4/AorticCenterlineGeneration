import os
import yaml
import torch
import random
import imgaug
import numpy as np

import logging as log

def batch_ids_generator(size, batch_size, shuffle=False):
    ids = np.arange(size)

    if shuffle:
        np.random.shuffle(ids)

    poses = np.arange(batch_size, size, batch_size)
    return np.split(ids, poses)

def init_determenistic(seed=1996, precision=10):
    """ NOTE options

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.determenistic = True

        may lead to numerical unstability
    """
    random.seed(seed)
    np.random.seed(seed)
    imgaug.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.determenistic = True
    torch.backends.cudnn.enabled = False

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_printoptions(precision=precision)

    os.environ['PYTHONHASHSEED'] = str(seed)

def init_logging(config, name, logtype='stream', **kwargs):
    config['LOGGER'] = log.getLogger(name)
    config['LOGGER'].setLevel(log.INFO)

    config['LOGGER'].handlers.clear()

    if logtype == 'stream':
        handler = log.StreamHandler()
    elif logtype == 'file':
        handler = log.FileHandler( kwargs.get('filename'),
                                   mode='a',
                                   encoding='utf-8' )
    else:
        handler = log.NullHandler()

    formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    config['LOGGER'].addHandler(handler)

def load_yaml_config(path):
    #assert path.is_file() #исправил

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    return data
