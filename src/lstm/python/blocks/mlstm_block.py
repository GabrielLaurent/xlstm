from src.lstm.python.blocks.lstm_block import LSTMBlock
import torch
import torch.nn as nn


class mLSTMBlock(LSTMBlock):
    """Implementation of the mLSTM block."""

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False, merge_mode='sum'):
        super(mLSTMBlock, self).__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
        self.merge_mode = merge_mode

        if merge_mode not in ['sum', 'mul', 'concat', 'none']:
            raise ValueError("Invalid merge mode specified: {}.  Must be one of ['sum', 'mul', 'concat', 'none']".format(merge_mode))

    def forward(self, x, hc=None):
        # Pass input to lstm

        output, (h_n, c_n) = self.lstm(x, hc)

        # Apply merge mode.
        if self.merge_mode == 'sum':
            merged_output = torch.sum(output, dim=1)
        elif self.merge_mode == 'mul':
            merged_output = torch.prod(output, dim=1)
        elif self.merge_mode == 'concat':
            merged_output = output.reshape(output.shape[0], -1)
        else:
            merged_output = output


        return merged_output, (h_n, c_n)
