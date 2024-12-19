import numpy as np

class LSTMBlock:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights (simplified for example)
        self.W_i = np.random.randn(hidden_size, input_size) * 0.01
        self.U_i = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_i = np.zeros((hidden_size, 1))

        self.W_f = np.random.randn(hidden_size, input_size) * 0.01
        self.U_f = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_f = np.zeros((hidden_size, 1))

        self.W_o = np.random.randn(hidden_size, input_size) * 0.01
        self.U_o = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_o = np.zeros((hidden_size, 1))

        self.W_g = np.random.randn(hidden_size, input_size) * 0.01
        self.U_g = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_g = np.zeros((hidden_size, 1))


    def forward(self, x, h_prev, c_prev):
        """
        Forward pass for a single LSTM block.

        Args:
            x (np.ndarray): Input vector of shape (input_size, 1)
            h_prev (np.ndarray): Previous hidden state of shape (hidden_size, 1)
            c_prev (np.ndarray): Previous cell state of shape (hidden_size, 1)

        Returns:
            tuple: (h_next, c_next)
        """
        i = sigmoid(np.dot(self.W_i, x) + np.dot(self.U_i, h_prev) + self.b_i)
        f = sigmoid(np.dot(self.W_f, x) + np.dot(self.U_f, h_prev) + self.b_f)
        o = sigmoid(np.dot(self.W_o, x) + np.dot(self.U_o, h_prev) + self.b_o)
        g = np.tanh(np.dot(self.W_g, x) + np.dot(self.U_g, h_prev) + self.b_g)

        c_next = f * c_prev + i * g
        h_next = o * np.tanh(c_next)

        return h_next, c_next


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
