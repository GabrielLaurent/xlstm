import unittest
import torch
from src.lstm.python.lstm import LSTMBlock

class TestLSTM(unittest.TestCase):
    def test_lstm_output_size(self):
        input_size = 10
        hidden_size = 20
        model = LSTMBlock(input_size, hidden_size)
        input_tensor = torch.randn(1, 5, input_size) # batch, sequence, features
        output = model(input_tensor)
        self.assertEqual(output.shape, torch.Size([1, 5, hidden_size]))

if __name__ == '__main__':
    unittest.main()