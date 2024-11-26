import pytest
import torch

from src.lstm.python.blocks.lstm_block import LSTMBlock
from src.lstm.python.blocks.mlstm_block import mLSTMBlock
from src.lstm.python.blocks.slstm_block import sLSTMBlock


@pytest.fixture
def lstm_input():
    # Create a sample input tensor for testing
    batch_size = 2
    seq_len = 10
    input_size = 32
    return torch.randn(batch_size, seq_len, input_size)

@pytest.fixture
def hidden_state(lstm_input):
    batch_size = lstm_input.size(0)
    hidden_size = 64 # Example hidden size, must match model
    return (torch.randn(batch_size, hidden_size), torch.randn(batch_size, hidden_size))

@pytest.mark.parametrize("block_type", [LSTMBlock, mLSTMBlock, sLSTMBlock])
def test_lstm_block_forward(lstm_input, hidden_state, block_type):
    input_size = lstm_input.size(2)
    hidden_size = 64
    lstm_block = block_type(input_size, hidden_size)

    output, new_hidden = lstm_block(lstm_input, hidden_state)

    # Assert that the output and hidden state have the correct shapes
    assert output.shape == lstm_input.shape
    assert isinstance(new_hidden, tuple)
    assert len(new_hidden) == 2
    assert new_hidden[0].shape == hidden_state[0].shape
    assert new_hidden[1].shape == hidden_state[1].shape

@pytest.mark.parametrize("block_type", [LSTMBlock, mLSTMBlock, sLSTMBlock])
def test_lstm_block_initial_hidden_state(lstm_input, hidden_state, block_type):
    input_size = lstm_input.size(2)
    hidden_size = 64
    lstm_block = block_type(input_size, hidden_size)

    # Test the initial hidden state
    batch_size = lstm_input.size(0)
    initial_hidden = lstm_block.init_hidden(batch_size)

    assert isinstance(initial_hidden, tuple)
    assert len(initial_hidden) == 2
    assert initial_hidden[0].shape == (batch_size, hidden_size)
    assert initial_hidden[1].shape == (batch_size, hidden_size)