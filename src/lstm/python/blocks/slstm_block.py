import torch
from torch import nn
from src.lstm.python.blocks.lstm_block import LSTMBlock

try:
    from src.lstm.cpp import slstm_cpp
    use_cpp = True
except ImportError:
    use_cpp = False
    print("Could not import C++ sLSTM implementation. Falling back to Python.")

class sLSTMBlock(LSTMBlock):
    def __init__(self, input_size, hidden_size):
        super().__init__(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)


    def forward(self, input, state = None):

        if state is None:
            h_0 = torch.zeros(1, self.hidden_size, dtype=input.dtype, device=input.device)
            c_0 = torch.zeros(1, self.hidden_size, dtype=input.dtype, device=input.device)
            state = (h_0, c_0)

        if use_cpp:
            input_list = input.tolist()[0]
            weights_ih_list = self.lstm_cell.weight_ih.tolist()
            weights_hh_list = self.lstm_cell.weight_hh.tolist()
            bias_ih_list = self.lstm_cell.bias_ih.tolist()
            bias_hh_list = self.lstm_cell.bias_hh.tolist()
            prev_h_list = state[0].tolist()[0]
            prev_c_list = state[1].tolist()[0]
            
            output = slstm_cpp.forward(
                input_list,
                weights_ih_list,
                weights_hh_list,
                bias_ih_list,
                bias_hh_list,
                self.input_size,
                self.hidden_size,
                prev_h_list,
                prev_c_list
            )
            h_next = torch.tensor([output])
            c_next = torch.rand(1, self.hidden_size, dtype=input.dtype, device=input.device)
            return h_next, (h_next, c_next)
        else:
            h_next, c_next = self.lstm_cell(input, state)
            return h_next, (h_next, c_next)
