import numpy as np

from network import Network

class QuantumNetwork(Network):

	def sigmoid(self, z: int) -> float:
		return 1.0/(1.0+np.exp(-z))
	
	def sigmoid_prime(self, z: int) -> float:
		return self.sigmoid(z)*(1-self.sigmoid(z))

	def backprop(self, x, y):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		# feedforward
		activation = x
		activations = [x] # stores activations layer by layer
		zs = [] # stores z vectors layer by layer
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = self.sigmoid(z)
			activations.append(activation)
	
		# backward pass
		delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
	
		for _layer in range(2, self.num_layers):
			z = zs[-_layer]
			sp = self.sigmoid_prime(z)
			delta = np.dot(self.weights[-_layer+1].transpose(), delta) * sp
			nabla_b[-_layer] = delta
			nabla_w[-_layer] = np.dot(delta, activations[-_layer-1].transpose())
		return (nabla_b, nabla_w)