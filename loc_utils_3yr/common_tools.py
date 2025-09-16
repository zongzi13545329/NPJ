
import numpy as np
import random
import os
import torch
import argparse
import easydict
import yaml
import sys

from loc_utils_3yr.model_util import Modality

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def input_dim(input_axis, freq_bands, channels):
    return input_axis * ((freq_bands * 2)+1) + channels


class YmlConfig(object):
    def __init__(self, yml_path) -> None:
        self.yml_path = yml_path
        with open(self.yml_path, 'r') as fin:
            self.obj = easydict.EasyDict(yaml.load(fin, yaml.Loader))
        
        
    def flush(self):
        with open(self.yml_path, 'r') as fin:
            self.obj = easydict.EasyDict(yaml.load(fin, yaml.Loader))
    
    def parse_to_modality(self, modality_dict: easydict.EasyDict):
        
        path = modality_dict.get('path', 'data')
        feature_dim = modality_dict.get('feature_dim', 100)
        modality_name = modality_dict.get('modality_name', 'img')

        return Modality(
            path=path,
            feature_dim=feature_dim,
            modality_name=modality_name
        )
        