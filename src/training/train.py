import torch
import torch.nn as nn
import torch.optim as optim
from src.lstm.python.slstm import StackedLSTM


def train(config, experiment_manager):
    # Assuming config is a dictionary containing training parameters
    num_epochs = config.get('training', {}).get('epochs', 10)
    learning_rate = config.get('training', {}).get('learning_rate', 0.001)
    input_size = config['model']['input_size']
    hidden_size = config['model']['hidden_size']
    output_size = config['model']['output_size']
    num_layers = config['model']['num_layers']

    # Dummy data and model initialization (replace with actual data loading)
    model = StackedLSTM(input_size, hidden_size, output_size, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Dummy training data
    input_tensor = torch.randn(10, 1, input_size)  # sequence_length, batch_size, input_size
    target_tensor = torch.randint(0, output_size, (10,))  # sequence_length

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
        experiment_manager.log(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
        
        # Save checkpoint every few epochs
        if (epoch + 1) % 5 == 0:
            experiment_manager.save_checkpoint(model, epoch + 1)
            
    return model