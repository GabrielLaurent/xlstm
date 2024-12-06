import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.config.config_loader import load_config
from src.components.dummy_data_loader import DummyDataLoader
from src.lstm.python.mlstm import mLSTM
from src.training.train import train_epoch
from src.training.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description='Train mLSTM model')
    parser.add_argument('--config', type=str, default='experiments/example_config.yaml', help='Path to the config file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and config['training']['use_cuda'] else 'cpu')

    # Dummy Data Loaders
    train_loader = DummyDataLoader(config['data']['train_size'], config['data']['input_size'], config['data']['sequence_length'])
    val_loader = DummyDataLoader(config['data']['val_size'], config['data']['input_size'], config['data']['sequence_length'])

    # Model Initialization
    input_size = config['data']['input_size']
    hidden_size = config['model']['hidden_size']
    num_layers = config['model']['num_layers']
    output_size = config['data']['input_size']  # Assuming output size is same as input size for simplicity, can be configured

    model = mLSTM(input_size, hidden_size, num_layers).to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Training Loop
    num_epochs = config['training']['num_epochs']
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    print('Finished Training')

if __name__ == "__main__":
    main()