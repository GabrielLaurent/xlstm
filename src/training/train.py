import torch
import torch.nn as nn
import torch.optim as optim
from src.lstm.python.lstm import LSTMBlock
from src.data.formal_language_data_generator import generate_formal_language_data

# Hyperparameters (move to config later)
input_size = 10
hidden_size = 20
num_epochs = 10
learning_rate = 0.001


def train():
    # Generate dummy input and target data
    num_samples = 100
    input_data = torch.randn(num_samples, 1, input_size)
    target_data = torch.randn(num_samples, 1, hidden_size)

    # Create LSTM model, loss function, and optimizer
    model = LSTMBlock(input_size, hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        outputs = model(input_data)
        loss = criterion(outputs, target_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    train()
