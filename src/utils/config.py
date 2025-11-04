import yaml
import os

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        # Return default config
        return {
            'paths': {
                'data_raw': 'data/raw/Ban-Sign-Sent-9K-V1',
                'data_processed': 'data/processed'
            }
        }

def save_config(config, config_path="config.yaml"):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)