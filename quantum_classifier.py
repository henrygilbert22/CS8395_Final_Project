"""
Please note that this class is heavily inspired from the following repo:
https://github.com/Qiskit/qiskit-machine-learning

While I wish I could claim I figured out to do this level of machine learning in conjuection
with quantum computation, I'm not a single course in quantum computing is quite enough :)

The is not copied, but I certainly could not have figured out the underlying methodology
without a deep dive into their source code. Additionally, inherenting from TrainableModel
is a bit of a hack, but it's the only way I could figure out how to use the QuantumCircuit
in this context.

"""

from qiskit_machine_learning.algorithms.trainable_model import TrainableModel
from qiskit.algorithms.optimizers import Optimizer

from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np

from quantum_network import QuantumNeuralNetwork

class QuantumClassifier(TrainableModel, ClassifierMixin):
	""" Wrapper class for the QuantumNeuralNetwork class. It helps abstract
		the undeleted QuantumCircuit from the NeuralNetwork class. Additionally,
		it makes the integration with the scipy optimizer cleaner. """

	_X: np.array
	_y: np.array

	_last_forward_weights: np.ndarray = None
	_last_forward: np.ndarray = None
	_num_samples: int

	def __init__(
		self,
		neural_network: QuantumNeuralNetwork,
		optimizer: Optimizer) -> None:
		""" Initialize a QuantumClassifier by setting the target_encoder and initializing the 
			parent TrainableModel.
			
		Args:
			neural_network: The neural network to use for the classifier.
			optimizer: The optimizer to use for the classifier.
			
		Returns:
			None
		"""
		
		super().__init__(
			neural_network, 
			'squared_error',    # loss Function
			optimizer, 
			False,              # Warm Start
			None,               # Initial Point
			None                # Callback
		)
		
		self._target_encoder = LabelEncoder()

	def learn(self, X: np.ndarray, y: np.ndarray) -> None: 
		""" Learn the classification function by training the underlying neural network
			through a scipy optimization.
			
		Args:
			X: The input data.
			y: The target labels.
		
		Returns:
			None
		"""
		
		self._X = X
		self._y = y
		self._num_samples = X.shape[0]
		
		self._fit_result = self._optimizer.minimize(
			fun=self._objective,
			x0=self._choose_initial_point(),
			jac=self._gradient,
		)

	def _forward(self, weights: np.ndarray) -> np.ndarray:
		""" Compute a forward pass of the neural network. Firstly check
			if the weights are the same as the last forward pass. If so,
			return the last forward pass. Otherwise, compute the forward pass
			and store it for future use.
			
		Args:
			weights: The weights to use for the forward pass.
		
		Returns:
			The forward pass of the neural network.
		"""
		
		
		if self._last_forward_weights is not None and np.all(np.isclose(weights, self._last_forward_weights)):
			return self._last_forward
		
		# Compute forward pass and store it for future use.
		self._last_forward = self._neural_network.forward(self._X, weights)
		
		# Must use a copy to avoid modifying the original weights.
		self._last_forward_weights = np.copy(weights)
		
		return self._last_forward

	def _objective(self, weights: np.array) -> float:
		""" Compute the objective function of the neural network.
		
		Args:
			weights: The weights to use for the forward pass.
	
		Returns:
			The objective function of the neural network.
		"""
		
		probs = self._forward(weights)
		num_outputs = self._neural_network.output_shape[0]
		
		val = 0.0
		num_samples = self._X.shape[0]
		
		for i in range(num_outputs):
			val += probs[:, i] @ self._loss(np.full(num_samples, i), self._y)
		
		return val / self._num_samples

	def _gradient(self, weights: np.ndarray) -> np.ndarray:
		""" Compute the gradient of the objective function of the neural network.
		
		Args:
			weights: The weights to use for the forward pass.
		
		Returns:
			The gradient of the objective function of the neural network.
		"""
		
		_, weight_prob_grad = self._neural_network.backward(self._X, weights)
		grad = np.zeros((1, self._neural_network.num_weights))
		
		num_samples = self._X.shape[0]
		num_outputs = self._neural_network.output_shape[0]
		
		for i in range(num_outputs):
			grad += weight_prob_grad[:, i, :].T @ self._loss(np.full(num_samples, i), self._y)

		return grad / self._num_samples

	def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
		""" Compute the accuracy of the neural network on the given data. This is the sole reason
			we've inherented ClassifierMixin. 
			
		Args:
			X: The input data.
			y: The target labels.
			sample_weight: The sample weights.
		
		Returns:
			The accuracy of the neural network on the given data.
		"""

		return ClassifierMixin.score(self, X, y, sample_weight)

	def predict(self, X: np.ndarray) -> np.ndarray: 
		""" Predict the labels of the given data.
		
		Args:
			X: The input data.
		
		Returns:
			The predicted labels.
		"""

		return np.argmax(self._neural_network.forward(X, self._fit_result.x), axis=1)
		