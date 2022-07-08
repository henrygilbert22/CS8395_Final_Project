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

	def compute_loss(self, X: np.array, y: list, regularization: float = 0.0) -> float:
		""" Compute the cross entropy loss for the network.
		
		Args:
			X: list of input data
			y: list of labels
		
		Returns:
			float of loss value
		"""

		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		N, D = X.shape

		network_output = self.forward_pass(X)

		loss = math_lib.cross_entropy_loss(network_output, y)
		loss += regularization * (np.sum(W1 * W1) + np.sum(W2 * W2))

	def forward_pass(self, X: np.array) -> np.array:
		""" Forward pass of the network.
		
		Args:
			X: list of input data
		
		Returns:
			list of output data
		"""

		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']

		layer_1_output = np.dot(X, W1) + b1
		layer_1_output = math_lib.ReLU(layer_1_output)
		layer_2_output = np.dot(layer_1_output, W2) + b2
		
		return layer_2_output, layer_1_output

	def backward_pass(self, layer_2_output: np.array, layer_1_output: np.array, 
		X: np.array, y: list, loss: float, learning_rate: float = 0.01, 
		regularization: float = 0.0) -> None:
		""" Forward pass of the network.
		
		Args:
			X: list of input data
			y: list of labels
			learning_rate: float
			regularization: float
		
		Returns:
			None
		"""

		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		N, _ = X.shape

		probabilities = self.softmax(layer_2_output)

		probabilities[(range(N), y)] -= 1
		dZ2 = probabilities / N

		dW2 = np.dot(layer_1_output.T, dZ2)
		db2 = np.ones((1, layer_1_output.shape[0])).dot(dZ2)

		dA1 = dZ2.dot(W2.T)
		dZ1 = dA1.copy()
		dZ1[layer_1_output <= 0] = 0
		dW1 = X.T.dot(dZ1)
		db1 = np.ones((1, N)).dot(dZ1)

		dW1 += 2 * regularization * W1
		dW2 += 2 * regularization * W2

		self.parameters['W1'] -= learning_rate * dW1
		self.parameters['b1'] -= learning_rate * db1[0]
		self.parameters['W2'] -= learning_rate * dW2
		self.parameters['b2'] -= learning_rate * db2[0]
