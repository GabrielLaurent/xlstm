import torch
import torch.nn as nn

class sLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(sLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gate
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        # Forget gate
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        # Output gate
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        # Cell gate
        self.W_g = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev, c_prev):
        # Concatenate input and previous hidden state
        combined = torch.cat((x, h_prev), dim=1)

        # Input gate
        i_t = torch.sigmoid(self.W_i(combined))
        # Forget gate
        f_t = torch.sigmoid(self.W_f(combined))
        # Output gate
        o_t = torch.sigmoid(self.W_o(combined))
        # Cell gate
        g_t = torch.tanh(self.W_g(combined))

        # Update cell state
        c_t = f_t * c_prev + i_t * g_t
        # Update hidden state
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

if __name__ == '__main__':
    # Example usage
    input_size = 10
    hidden_size = 20
    batch_size = 32

    # Create an instance of the sLSTM block
    slstm_block = sLSTMBlock(input_size, hidden_size)

    # Create dummy input, previous hidden state, and previous cell state
    x = torch.randn(batch_size, input_size)
    h_prev = torch.randn(batch_size, hidden_size)
    c_prev = torch.randn(batch_size, hidden_size)

    # Perform the forward pass
    h_t, c_t = slstm_block(x, h_prev, c_prev)

    # Print the shapes of the output
    print("Shape of h_t:", h_t.shape)
    print("Shape of c_t:", c_t.shape)