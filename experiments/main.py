import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from src.config.config_loader import load_config
from src.components.dummy_data_loader import DummyDataLoader  # Corrected import
from src.lstm.python.lstm import LSTM  # example model

def main():
    # 1. Load configuration
    config = load_config('experiments/example_config.yaml')

    # 2. Initialize model
    input_size = config['model']['input_size']
    hidden_size = config['model']['hidden_size']
    num_layers = config['model']['num_layers']
    output_size = config['model']['output_size']
    model_type = config['model']['type']  # Read the model type from the config

    if model_type == 'LSTM':  # Example model initialization
        model = LSTM(input_size, hidden_size, num_layers, output_size)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # 3. Load data (using dummy data loader)
    data_loader = DummyDataLoader(
        sequence_length=config['data']['sequence_length'],
        vocab_size=config['data']['vocab_size'],
        batch_size=config['training']['batch_size']
    )

    # 4. Training loop
    num_epochs = config['training']['epochs']
    learning_rate = config['training']['learning_rate']
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        inputs, targets = data_loader.get_batch()
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs.view(-1, output_size), targets.view(-1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # 5. Log statistics
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    main()
