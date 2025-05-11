#utils

import yaml
import random
import numpy as np

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed):
    """Ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except ImportError:
        pass  #torch may not be installed during static testing

def get_action_mapping():
    """Returns the action ID to label mapping."""
    return {
        0: 'Buy',
        1: 'Sell',
        2: 'Hold'
    }
