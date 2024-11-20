#!/bin/bash

# Navigate to the project root
cd /app

# Activate the virtual environment (if applicable)
# source venv/bin/activate

# Run the experiment using the provided config file
python experiments/main.py "$1"

# Exit script with code 0 if no errors, otherwise with error code.
exit 0