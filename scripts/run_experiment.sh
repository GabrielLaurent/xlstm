#!/bin/bash

# Example script to run an experiment

CONFIG_FILE="experiments/example_config.yaml"

python src/training/train.py --config $CONFIG_FILE
