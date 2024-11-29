import random

class FormalLanguageDataGenerator:
    def __init__(self, alphabet_size, sequence_length, grammar):
        self.alphabet_size = alphabet_size
        self.sequence_length = sequence_length
        self.grammar = grammar
        self.alphabet = [chr(i + ord('a')) for i in range(alphabet_size)]

    def generate_sequence(self): # Generates a sequence according to a grammar. Current implementation ignores grammar.
        return ''.join(random.choice(self.alphabet) for _ in range(self.sequence_length))

    def generate_data(self, num_samples):
        data = [self.generate_sequence() for _ in range(num_samples)]
        labels = [1] * num_samples # Placeholder labels (all positive)
        return data, labels


if __name__ == '__main__':
    # Example usage
    grammar = {}
    generator = FormalLanguageDataGenerator(alphabet_size=3, sequence_length=10, grammar=grammar)
    data, labels = generator.generate_data(num_samples=5)

    for i in range(len(data)):
        print(f"Sequence: {data[i]}, Label: {labels[i]}")
