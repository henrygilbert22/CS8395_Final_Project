import numpy as np

import math_lib

class QuantumNetwork():

    parameters: dict
    input_size: int
    output_size: int
    hidden_size: int

    def __init__(self, input_size: int, output_size: int, hidden_size: int) -> None:
        """ Initialize the network by setting the input_size, output_size, and hidden_size
        and initializing the weights and biases.
        
        Args:
            input_size: int
            output_size: int
            hidden_size: int
        
        Returns:
            None
        """
        
        self.parameters = {}
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.initialize_network()

    def initialize_network(self) -> None:
        """ Initialize the weights and biases of the network.
        
        Args:
            None
        
        Returns:
            None
        """

        self.parameters['W1'] = np.random.randn(self.input_size, self.hidden_size) * 1e-4
        self.parameters['b1'] = np.zeros(self.hidden_size)
        self.parameters['W2'] = np.random.randn(self.hidden_size, self.output_size) * 1e-4
        self.parameters['b2'] = np.zeros( self.output_size)

    