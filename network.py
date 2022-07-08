import numpy as np

class QuantumNetwork():

    parameters: dict
    input_size: int
    output_size: int
    hidden_size: int

    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        
        self.parameters = {}
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.initialize_network()

    def initialize_network(self):

        self.parameters['W1'] = np.random.randn(self.input_size, self.hidden_size) * 1e-4
        self.parameters['b1'] = np.zeros(self.hidden_size)
        self.parameters['W2'] = np.random.randn(self.hidden_size, self.output_size) * 1e-4
        self.parameters['b2'] = np.zeros( self.output_size)

    