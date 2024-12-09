import torch
import torch.nn as nn
import torch.optim as optim
from src.components.dummy_data_loader import DummyDataLoader
from src.lstm.python.lstm import LSTM  # Or any other LSTM implementation
from src.training.evaluate import evaluate


def train(model, data_loader, criterion, optimizer, epochs, validation_data_loader):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(data_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0
        # Evaluate after each epoch
        val_loss, val_accuracy = evaluate(model, validation_data_loader, criterion)
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')


if __name__ == '__main__':
    # Hyperparameters
    input_size = 10
    hidden_size = 20
    output_size = 5
    num_layers = 1
    learning_rate = 0.01
    epochs = 5
    batch_size = 32

    # Model
    model = LSTM(input_size, hidden_size, output_size, num_layers)

    # Data Loader
    data_loader = DummyDataLoader(input_size, output_size, sequence_length=20, num_sequences=100, batch_size=batch_size)
    validation_data_loader = DummyDataLoader(input_size, output_size, sequence_length=20, num_sequences=50, batch_size=batch_size)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    train(model, data_loader, criterion, optimizer, epochs, validation_data_loader)