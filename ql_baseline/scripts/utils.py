#scripts/utils.py

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

def get_action_mapping():
    """Returns the action ID to label mapping."""
    return {
        0: 'Buy',
        1: 'Sell',
        2: 'Hold'
    }

def epsilon_greedy_action(q_values, epsilon):
    """
    Chooses an action using epsilon-greedy strategy.
    Args:
        q_values (list): Q-values for available actions.
        epsilon (float): Probability of random action.
    Returns:
        int: selected action index.
    """
    if random.random() < epsilon:
        return random.randint(0, len(q_values) - 1)
    else:
        return np.argmax(q_values)

