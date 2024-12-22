import numpy as np

class DummyDataLoader:
    def __init__(self, input_dim, output_dim, num_samples=100):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_samples = num_samples

    def generate_data(self):
        inputs = np.random.rand(self.num_samples, self.input_dim)
        outputs = np.random.rand(self.num_samples, self.output_dim)
        return inputs, outputs

if __name__ == '__main__':
    # Example Usage
    data_loader = DummyDataLoader(input_dim=10, output_dim=5, num_samples=200)
    inputs, outputs = data_loader.generate_data()
    print("Input shape:", inputs.shape)
    print("Output shape:", outputs.shape)