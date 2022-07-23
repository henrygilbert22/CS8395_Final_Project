"""
Please note that this class is heavily inspired from the following repo:
https://github.com/Qiskit/qiskit-machine-learning

While I wish I could claim I figured out to do this level of machine learning in conjuection
with quantum computation, I'm not a single course in quantum computing is quite enough :)

The is not copied, but I certainly could not have figured out the underlying methodology
without a deep dive into their source code. Additionally, inherenting from SamplingNeuralNetwork
is a bit of a hack, but it's the only way I could figure out how to use the QuantumCircuit
in this context.

Specifically, the _probability_gradients method is explicitly taken from the source code.
I've refactored it a bit to make it more readable and to fit python coding conventions;
however, the core logic remains the same and is not mine.
"""

from qiskit_machine_learning.neural_networks.sampling_neural_network import SamplingNeuralNetwork
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import Gradient, CircuitSampler, StateFn, OperatorBase
from qiskit.utils import QuantumInstance

from typing import List, Union, Tuple, Callable, cast, Iterable
import numpy as np
from numbers import Integral
from scipy.sparse import coo_matrix

class QuantumNeuralNetwork(SamplingNeuralNetwork):
	""" This is base class to be used in the classifier. It abstracts out the lower
		level details of computing gradients, probabilities and running the actual
		quantum computations. """

	_input_params: List[Parameter]
	_weight_params: List[Parameter]
	_input_gradients: bool

	_gradient_circuit: OperatorBase
	_circuit: QuantumCircuit
	_circuit_transpiled: bool

	_original_output_shape: Union[int, Tuple[int, ...]]
	_output_shape: Union[int, Tuple[int, ...]]
	_interpret: Callable[[int], Union[int, Tuple[int, ...]]]
	_quantum_instance: QuantumInstance

	def __init__(
		self,
		circuit: QuantumCircuit,
		input_params: List[Parameter],
		weight_params: List[Parameter],
		interpret: Callable[[int], Union[int, Tuple[int, ...]]],
		output_shape: Union[int, Tuple[int, ...]],
		quantum_instance: QuantumInstance,
		input_gradients: bool = False,
		) -> None:
		""" Initializes the input and weight parameters, creates the quantum circuit, 
			quanutm gradient and gradient circuit. Additionally, initializes the parent
			class of SamplingNeuralNetwork.
			
		Args:
			circuit: The quantum circuit to be used for the neural network.
			input_params: The input parameters of the neural network.
			weight_params: The weight parameters of the neural network.
			interpret: A function to be used to interpret the output of the neural network.
			output_shape: The shape of the output of the neural network.
			quantum_instance: The quantum instance to be used for the neural network.
			input_gradients: Whether to compute the input gradients.
		
		Returns:
			None
		"""
		
		
		self._input_params = list(input_params or [])
		self._weight_params = list(weight_params or [])
		self._input_gradients = input_gradients

		self._circuit = circuit.copy()
		self._circuit_transpiled = False
		
		self._original_output_shape = output_shape
		self._output_shape = output_shape
		
		self._interpret = interpret
		self._quantum_instance = quantum_instance
		self._set_quantum_instance()

		super().__init__(
			len(self._input_params),
			len(self._weight_params),
			False,  # sparse
			False,  # sampling
			self._output_shape,
			self._input_gradients,
		)

		self._original_circuit = circuit
		self._gradient = Gradient()
		
		grad_circuit = self._original_circuit.copy()
		grad_circuit.remove_final_measurements()
		
		self._gradient_circuit = self._gradient.convert(
			StateFn(grad_circuit), 
			self._weight_params)
		
	def _set_quantum_instance(self) -> None:
		""" Sets the quantum instance for the neural network and 
			transpiles the quantum circuit if necessary. 
			
		Args:
			None
		
		Returns:
			None
		"""
		
		# add measurements if necessary
		if self._quantum_instance.is_statevector:
			if len(self._circuit.clbits) > 0:
				self._circuit.remove_final_measurements()
		
		elif len(self._circuit.clbits) == 0:
			self._circuit.measure_all()

		self._original_output_shape = self._output_shape
		self._original_interpret = self._interpret
		self._sampler = CircuitSampler(self._quantum_instance, param_qobj=False, caching="all")

		# Setting the interpret function, just return the input if no function is given
		self._interpret = self._interpret if self._interpret is not None else lambda x: x

		# transpile the QNN circuit to the quantum instance
		self._circuit = self._quantum_instance.transpile(
			self._circuit, 
			pass_manager=self._quantum_instance.unbound_pass_manager)[0]
		
		self._circuit_transpiled = True
		
	def _sample(self, input_data:np.ndarray, weights: np.ndarray) -> np.ndarray:
		""" Samples the quantum circuit for the given input data and weights
			by deriving and updating the weight parameters for the internal
			quantum circuit.
		
		Args:
			input_data: The input data to be used for the sampling.
			weights: The weights to be used for the sampling.
			
		Returns:
			The sampled output.
		"""

		orig_memory = self._quantum_instance.backend_options.get("memory")
		self._quantum_instance.backend_options["memory"] = True
		circuits = []
		num_samples = input_data.shape[0]
		
		for i in range(num_samples):
			param_values = {input_param: input_data[i, j] for j, input_param in enumerate(self._input_params)}
			param_values.update({weight_param: weights[j] for j, weight_param in enumerate(self._weight_params)})
			circuits.append(self._circuit.bind_parameters(param_values))

		if self._quantum_instance.bound_pass_manager is not None:
			circuits = self._quantum_instance.transpile(circuits, pass_manager=self._quantum_instance.bound_pass_manager)

		result = self._quantum_instance.execute(circuits, had_transpiled=self._circuit_transpiled)
		self._quantum_instance.backend_options["memory"] = orig_memory
		
		samples = np.zeros((num_samples, *self._output_shape))
		for i, circuit in enumerate(circuits):
			
			memory = result.get_memory(circuit)
			for j, b in enumerate(memory):
				samples[i, j, :] = self._interpret(int(b, 2))
		
		return samples
		
	def _probabilities(self, input_data: np.ndarray, weights: np.ndarray) -> np.ndarray:
		""" Samples the quantum circuit for the given input data and weights 
			and returns the probabilities of the output.
			
		Args:
			input_data: The input data to be used for the sampling.
			weights: The weights to be used for the sampling.
		
		Returns:
			The probabilities of the output.
		"""
		
		circuits = []
		num_samples = input_data.shape[0]
		
		for i in range(num_samples):
			
			# Binding the parameteres and adding the measurements
			param_values = {input_param: input_data[i, j] for j, input_param in enumerate(self._input_params)}
			param_values.update({weight_param: weights[j] for j, weight_param in enumerate(self._weight_params)})
			circuits.append(self._circuit.bind_parameters(param_values))

		# Transpiling the circuits if necessary
		if self._quantum_instance.bound_pass_manager is not None:
			circuits = self._quantum_instance.transpile(circuits, pass_manager=self._quantum_instance.bound_pass_manager)

		result = self._quantum_instance.execute(circuits, had_transpiled=self._circuit_transpiled)
		prob = np.zeros((num_samples, *self._output_shape))

		for i, circuit in enumerate(circuits):
			
			counts = result.get_counts(circuit)
			shots = sum(counts.values())

			# Evaluating the probabilities
			for b, v in counts.items():
				
				key = self._interpret(int(b, 2))
				if isinstance(key, Integral):
					key = (cast(int, key),)
				
				key = (i, *key)
				prob[key] += v / shots

		return prob    

	def _probability_gradients(self, input_data: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		""" Samples the quantum circuit for the given input data and weights
			and returns the gradients of the probabilities of the output.
			
		Args:
			input_data: The input data to be used for the sampling.
			weights: The weights to be used for the sampling.
		
		Returns:
			The gradients of the probabilities of the output.
		"""

		# check whether gradient circuit could be constructed
		if self._gradient_circuit is None:
			return None, None

		num_samples = input_data.shape[0]
		input_grad = None  # Initially, there are no input gradients
		weights_grad = np.zeros((num_samples, *self._output_shape, self._num_weights))
		param_values = {input_param: input_data[:, j] for j, input_param in enumerate(self._input_params)}
		param_values.update({weight_param: np.full(num_samples, weights[j]) for j, weight_param in enumerate(self._weight_params)})
		converted_op = self._sampler.convert(self._gradient_circuit, param_values)

		if self._input_gradients:
			input_grad = np.zeros((num_samples, *self._output_shape, self._num_inputs))
		
		if len(converted_op.parameters) > 0:
			
			# Each element corresponds to the gradient of the probability of the corresponding output
			param_bindings = [
				{param: param_values[i] for param, param_values in param_values.items()}
				for i in range(num_samples)
			]

			# Loop through gradient vectors and bind the remaining parameters
			grad = []
			for g_i, param_i in zip(converted_op, param_bindings):
				grad.append(g_i.bind_parameters(param_i).eval())
		
		else:
			grad = converted_op.eval()

		if self._input_gradients:
			num_grad_vars = self._num_inputs + self._num_weights
		else:
			num_grad_vars = self._num_weights

		# This will construct the gradients for each given sample
		for sample in range(num_samples):
			
			for i in range(num_grad_vars):
				coo_grad = coo_matrix(grad[sample][i]) 

				if self._input_gradients:
					grad_index = i if i < self._num_inputs else i - self._num_inputs
				
				else:
					grad_index = i

				for _, k, val in zip(coo_grad.row, coo_grad.col, coo_grad.data):
					
					# Run key through internal interpretation function
					key = self._interpret(k)
					if isinstance(key, Integral):
						key = (sample, int(key), grad_index)
					
					else:
						# If key is unhasable, then we must cast it
						key = tuple(cast(Iterable[int], key))
						key = (sample, *key, grad_index)

					
					if self._input_gradients:
						
						if i < self._num_inputs:
							input_grad[key] += np.real(val)
						else:
							weights_grad[key] += np.real(val)
					
					else:
						weights_grad[key] += np.real(val)
		
		return input_grad, weights_grad
	