from matplotlib import mathtext
import numpy as np
from collections import defaultdict

import math_lib

class BaseNetwork():

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

		W1, b1 = self.parameters['W1'], self.parameters['b1']
		W2, b2 = self.parameters['W2'], self.parameters['b2']
		N, D = X.shape

		layer_2_output, layer_1_output = self.forward_pass(X)

		loss = math_lib.cross_entropy_loss(layer_2_output, y)
		loss += regularization * (np.sum(W1 * W1) + np.sum(W2 * W2))

		return loss, layer_2_output, layer_1_output

	def forward_pass(self, X: np.array) -> np.array:
		""" Forward pass of the network.
		
		Args:
			X: list of input data
		
		Returns:
			list of output data
		"""

		W1, b1 = self.parameters['W1'], self.parameters['b1']
		W2, b2 = self.parameters['W2'], self.parameters['b2']

		layer_1_output = np.dot(X, W1) + b1
		layer_1_output = math_lib.ReLU(layer_1_output)
		layer_2_output = np.dot(layer_1_output, W2) + b2
		
		return layer_2_output, layer_1_output

	def compute_backward_prop(self, layer_2_output: np.array, layer_1_output: np.array, 
		X: np.array, y: list, learning_rate: float, 
		regularization: float) -> None:
		""" Forward pass of the network.
		
		Args:
			X: list of input data
			y: list of labels
			learning_rate: float
			regularization: float
		
		Returns:
			None
		"""

		W1, b1 = self.parameters['W1'], self.parameters['b1']
		W2, b2 = self.parameters['W2'], self.parameters['b2']
		N, _ = X.shape

		probabilities = math_lib.softmax_activation(layer_2_output)

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

	def predict(self, X: np.array) -> list:
		""" Predict the labels of the input data.

		Args:
			X: list of input data
		
		Returns:
			list of predicted labels
		"""

        
		y_pred = [np.argmax(layer_2) for  layer_2 in self.forward_pass(X)[0]]
		return np.array(y_pred)


	def train(self, X_train, y_train, X_val, y_val,
				learning_rate=1e-3,
				reg=5e-6, num_epochs=100,
				batch_size=200, learning_rate_decay=0.95,
				early_stopping=False, patience=3) -> None:
		""" Train the network using stochastic gradient descent. Will update the weights and biases of the network
			to the best found network based off the validation set.
		
		Args:
			X: list of input data
			y: list of labels
			learning_rate: float
			learning_rate_decay: float
			reg: float
			num_iters: int
			batch_size: int
			verbose: bool
		
		Returns:
			None
		"""

		train_size = X_train.shape[0]
		iterations_per_epoch = max(train_size / batch_size, 1)
		training_metrics = defaultdict(list)
		
		best_val_acc = 0
		best_parameters = {}
		num_increasing_epochs = 0

		for it in range(int(num_epochs*iterations_per_epoch)):

			X_batch = X_train[np.random.choice(train_size, batch_size, replace=True)]	
			y_batch = y_train[np.random.choice(train_size, batch_size, replace=True)]

			loss, layer_2_output, layer_1_output = self.compute_loss(X_batch, y_batch, reg)

			self.compute_backward_prop(layer_2_output, layer_1_output, X_batch, y_batch, learning_rate, reg)

			if it % iterations_per_epoch == 0:

				train_acc = (self.predict(X_batch) == y_batch).mean()
				val_acc = (self.predict(X_val) == y_val).mean()
				training_metrics['loss_history'].append(loss)
				training_metrics['train_acc_history'].append(train_acc)
				training_metrics['val_acc_history'].append(val_acc)

				self.pretty_print_training_metrics(training_metrics, it/iterations_per_epoch)
				learning_rate *= learning_rate_decay

				if early_stopping:
					
					if val_acc > best_val_acc:

						best_val_acc = val_acc
						num_increasing_epochs = 0
						for k, v in self.parameters.items():
							best_parameters[k] = v.copy()

					else:

						num_increasing_epochs += 1
						if num_increasing_epochs == patience:

							self.parameters = best_parameters
							return training_metrics

					


		self.parameters = best_parameters
		return training_metrics

	def pretty_print_training_metrics(self, training_metrics: dict, it: int):

		print(f"*************** ITERATION: {it} ***************")
		print(f"		Loss: {training_metrics['loss_history'][-1]}")
		print(f"		Train Accuracy: {training_metrics['train_acc_history'][-1]}")
		print(f"		Validation Accuracy: {training_metrics['val_acc_history'][-1]}")
		print(f"***********************************************")