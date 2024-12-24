import yaml
import json


def load_config(config_path: str) -> dict:
    """Loads configuration from a YAML or JSON file.

    Args:
        config_path: The path to the configuration file.

    Returns:
        A dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file format is invalid.
    """
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config = json.load(f)
            else:
                raise ValueError("Unsupported file format. Only YAML and JSON are supported.")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in {config_path}: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {config_path}: {e}")


if __name__ == '__main__':
    # Example usage (assuming you have a config.yaml file)
    try:
        config = load_config('experiments/example_config.yaml')
        print("Configuration loaded successfully:")
        print(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading configuration: {e}")