import torch
import torch.nn as nn
import numpy as np
import random
from .device import *
from . import feature
from .feature import NUM_ATOM_FEATURES

def initialize_model(model, device, load_save_file=None):
    """
    To set device of model
    If there is save_file for model, load it
    """
    if load_save_file:
        model.load_state_dict(torch.load(load_save_file), strict=False) 
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                #nn.init.normal(param, 0.0, 0.15)
                nn.init.xavier_normal_(param)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    return model
