import pytest
import yaml
from src.config.config_loader import ConfigLoader


def test_config_loader_valid_yaml():
    # Create a temporary YAML file for testing
    config_data = {
        'model': {
            'type': 'LSTM',
            'hidden_size': 128
        },
        'training': {
            'epochs': 10,
            'learning_rate': 0.001
        }
    }
    with open('temp_config.yaml', 'w') as f:
        yaml.dump(config_data, f)

    # Load the configuration using ConfigLoader
    config = ConfigLoader.load('temp_config.yaml')

    # Assert that the configuration values are loaded correctly
    assert config.model.type == 'LSTM'
    assert config.model.hidden_size == 128
    assert config.training.epochs == 10
    assert config.training.learning_rate == 0.001

    # Clean up the temporary file
    import os
    os.remove('temp_config.yaml')


def test_config_loader_invalid_yaml():
    # Create a temporary invalid YAML file for testing
    with open('temp_config.yaml', 'w') as f:
        f.write('This is not a valid YAML file.')

    # Assert that ConfigLoader raises an exception when loading the invalid file
    with pytest.raises(yaml.YAMLError):
        ConfigLoader.load('temp_config.yaml')

    # Clean up the temporary file
    import os
    os.remove('temp_config.yaml')