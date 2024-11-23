import argparse
import yaml
import os
import datetime
import torch
from src.config.config_loader import load_config
from src.training.train import train
from src.training.evaluate import evaluate
from experiments.utils.experiment_manager import ExperimentManager


def main():
    parser = argparse.ArgumentParser(description='Run experiments for XLSTM models.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--experiment_name', type=str, default=None, help='Name of the experiment. If none, a timestamped name will be used.')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.experiment_name is None:
        experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        experiment_name = args.experiment_name

    experiment_manager = ExperimentManager(config, experiment_name)
    experiment_manager.setup_experiment()

    # Save config to experiment directory
    experiment_manager.save_config(args.config)
    
    # Training
    model = train(config, experiment_manager)

    # Evaluation
    evaluate(model, config, experiment_manager)

    print(f"Experiment completed. Results saved in: {experiment_manager.experiment_dir}")


if __name__ == "__main__":
    main()