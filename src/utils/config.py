import yaml
import os

def load_config(config_path=\"config.yaml\"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, config_path=\"config.yaml\"):
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
