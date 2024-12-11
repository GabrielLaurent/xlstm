import torch

class DummyDataLoader:
    def __init__(self, sequence_length, vocab_size, batch_size):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size

    def get_batch(self):
        # Generate random input sequences and targets
        inputs = torch.randint(0, self.vocab_size, (self.batch_size, self.sequence_length))
        targets = torch.randint(0, self.vocab_size, (self.batch_size, self.sequence_length))
        return inputs, targets