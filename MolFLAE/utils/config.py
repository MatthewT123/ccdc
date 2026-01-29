import yaml


def load_config(config_path):
    """Load in a .yaml config file from the specified path"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config
