from qiskit_machine_learning.neural_networks.sampling_neural_network import SamplingNeuralNetwork
from qiskit_machine_learning.algorithms.trainable_model import TrainableModel
from qiskit_machine_learning.neural_networks import NeuralNetwork
from qiskit_machine_learning.utils.loss_functions import Loss

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import Gradient, CircuitSampler, StateFn, OpflowError, OperatorBase
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import Optimizer

from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from typing import List, Union, Tuple, Callable, cast, Iterable
import numpy as np
from numbers import Integral
from scipy.sparse import coo_matrix

class QNN(SamplingNeuralNetwork):
    
    _input_params: List[Parameter]
    _weight_params: List[Parameter]
    _input_gradients: bool
    
    _gradient_circuit: OperatorBase
    _circuit: QuantumCircuit
    
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
        """
        
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
       
        # add measurements in case none are given
        if self._quantum_instance.is_statevector:
            if len(self._circuit.clbits) > 0:
                self._circuit.remove_final_measurements()
        
        elif len(self._circuit.clbits) == 0:
            self._circuit.measure_all()

        # set interpret and compute output shape
        # save original values
        self._original_output_shape = self._output_shape
        self._original_interpret = self._interpret

        # derive target values to be used in computations
        self._interpret = self._interpret if self._interpret is not None else lambda x: x

        # prepare sampler
        self._sampler = CircuitSampler(self._quantum_instance, param_qobj=False, caching="all")
        
        # transpile the QNN circuit
        self._circuit = self._quantum_instance.transpile(
            self._circuit, 
            pass_manager=self._quantum_instance.unbound_pass_manager)[0]
        self._circuit_transpiled = True
        
    def _sample(self, input_data:np.ndarray, weights: np.ndarray) -> np.ndarray:

        # evaluate operator
        orig_memory = self._quantum_instance.backend_options.get("memory")
        self._quantum_instance.backend_options["memory"] = True

        circuits = []
        # iterate over samples, each sample is an element of a batch
        num_samples = input_data.shape[0]
        
        for i in range(num_samples):
            
            param_values = {
                input_param: input_data[i, j] for j, input_param in enumerate(self._input_params)
            }
            
            param_values.update(
                {weight_param: weights[j] for j, weight_param in enumerate(self._weight_params)}
            )
            
            circuits.append(self._circuit.bind_parameters(param_values))

        if self._quantum_instance.bound_pass_manager is not None:
            circuits = self._quantum_instance.transpile(
                circuits, pass_manager=self._quantum_instance.bound_pass_manager
            )

        result = self._quantum_instance.execute(circuits, had_transpiled=self._circuit_transpiled)
        # set the memory setting back
        self._quantum_instance.backend_options["memory"] = orig_memory

        # return samples
        samples = np.zeros((num_samples, *self._output_shape))
        # collect them from all executed circuits
        
        for i, circuit in enumerate(circuits):
            
            memory = result.get_memory(circuit)
            for j, b in enumerate(memory):
                samples[i, j, :] = self._interpret(int(b, 2))
       
        return samples
       
    def _probabilities(self, input_data: np.ndarray, weights: np.ndarray) -> np.ndarray:
        
        # evaluate operator
        circuits = []
        num_samples = input_data.shape[0]
        
        for i in range(num_samples):
           
            param_values = {
                input_param: input_data[i, j] for j, input_param in enumerate(self._input_params)
            }
           
            param_values.update(
                {weight_param: weights[j] for j, weight_param in enumerate(self._weight_params)}
            )
            
            circuits.append(self._circuit.bind_parameters(param_values))

        if self._quantum_instance.bound_pass_manager is not None:
            circuits = self._quantum_instance.transpile(
                circuits, pass_manager=self._quantum_instance.bound_pass_manager
            )

        result = self._quantum_instance.execute(circuits, had_transpiled=self._circuit_transpiled)
        # initialize probabilities
        prob = np.zeros((num_samples, *self._output_shape))

        for i, circuit in enumerate(circuits):
            
            counts = result.get_counts(circuit)
            shots = sum(counts.values())

            # evaluate probabilities
            for b, v in counts.items():
                
                key = self._interpret(int(b, 2))
                if isinstance(key, Integral):
                    key = (cast(int, key),)
               
                key = (i, *key)  # type: ignore
                prob[key] += v / shots

        return prob    

    def _probability_gradients(self, input_data: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        # check whether gradient circuit could be constructed
        if self._gradient_circuit is None:
            return None, None

        num_samples = input_data.shape[0]

        # initialize empty gradients
        input_grad = None  # by default we don't have data gradients
        
        if self._input_gradients:
            input_grad = np.zeros((num_samples, *self._output_shape, self._num_inputs))
        
        weights_grad = np.zeros((num_samples, *self._output_shape, self._num_weights))

        param_values = {
            input_param: input_data[:, j] for j, input_param in enumerate(self._input_params)
        }
        
        param_values.update(
            {
                weight_param: np.full(num_samples, weights[j])
                for j, weight_param in enumerate(self._weight_params)
            }
        )

        converted_op = self._sampler.convert(self._gradient_circuit, param_values)
        
        # if statement is a workaround for https://github.com/Qiskit/qiskit-terra/issues/7608
        if len(converted_op.parameters) > 0:
           
            # create an list of parameter bindings, each element corresponds to a sample in the dataset
            param_bindings = [
                {param: param_values[i] for param, param_values in param_values.items()}
                for i in range(num_samples)
            ]

            grad = []
            # iterate over gradient vectors and bind the correct leftover parameters
            for g_i, param_i in zip(converted_op, param_bindings):
                # bind or re-bind remaining values and evaluate the gradient
                grad.append(g_i.bind_parameters(param_i).eval())
       
        else:
            grad = converted_op.eval()

        if self._input_gradients:
            num_grad_vars = self._num_inputs + self._num_weights
        else:
            num_grad_vars = self._num_weights

        # construct gradients
        for sample in range(num_samples):
            
            for i in range(num_grad_vars):
                coo_grad = coo_matrix(grad[sample][i])  # this works for sparse and dense case

                # get index for input or weights gradients
                if self._input_gradients:
                    grad_index = i if i < self._num_inputs else i - self._num_inputs
                
                else:
                    grad_index = i

                for _, k, val in zip(coo_grad.row, coo_grad.col, coo_grad.data):
                    # interpret integer and construct key
                    
                    key = self._interpret(k)
                    if isinstance(key, Integral):
                        key = (sample, int(key), grad_index)
                    
                    else:
                        # if key is an array-type, cast to hashable tuple
                        key = tuple(cast(Iterable[int], key))
                        key = (sample, *key, grad_index)

                    # store value for inputs or weights gradients
                    if self._input_gradients:
                        # we compute input gradients first
                        if i < self._num_inputs:
                            input_grad[key] += np.real(val)
                        else:
                            weights_grad[key] += np.real(val)
                    else:
                        weights_grad[key] += np.real(val)
        # end of for each sample
        return input_grad, weights_grad
      
class QNNClassifier(TrainableModel, ClassifierMixin):
    
    _X: np.array
    _y: np.array
    
    _last_forward_weights: np.ndarray = None
    _last_forward: np.ndarray = None
    _num_samples: int
    
    def __init__(
        self,
        neural_network: QNN,
        optimizer: Optimizer,
    ):
        
        super().__init__(
            neural_network, 
            'squared_error',    # loss Function
            optimizer, 
            False,              # Warm Start
            None,               # Initial Point
            None                # Callback
        )
        
        self._target_encoder = LabelEncoder()
    
    def learn(self, X: np.ndarray, y: np.ndarray): 
        
        self._X = X
        self._y = y
        self._num_samples = X.shape[0]
        
        self._fit_result = self._optimizer.minimize(
            fun=self._objective,
            x0=self._choose_initial_point(),
            jac=self._gradient,
        )
        return self
    
    def _forward(self, weights: np.ndarray) -> np.ndarray:
       
        # if we get the same weights, we don't compute the forward pass again.
        
        if self._last_forward_weights is None or (
            not np.all(np.isclose(weights, self._last_forward_weights))
        ):
            # compute forward and cache the results for re-use in backward
            self._last_forward = self._neural_network.forward(self._X, weights)
            # a copy avoids keeping a reference to the same array, so we are sure we have
            # different arrays on the next iteration.
            self._last_forward_weights = np.copy(weights)
        
        return self._last_forward
    
    def _objective(self, weights) -> float:
        
        probs = self._forward(weights)

        num_outputs = self._neural_network.output_shape[0]
        val = 0.0
        num_samples = self._X.shape[0]
        
        for i in range(num_outputs):
            # for each output we compute a dot product of probabilities of this output and a loss
            # vector.
            # loss vector is a loss of a particular output value(value of i) versus true labels.
            # we do this across all samples.
            val += probs[:, i] @ self._loss(np.full(num_samples, i), self._y)
        
        val = val / self._num_samples
        return val
    
    def _gradient(self, weights: np.ndarray) -> np.ndarray:
        
        # weight probability gradient is of shape (N, num_outputs, num_weights)
        _, weight_prob_grad = self._neural_network.backward(self._X, weights)

        grad = np.zeros((1, self._neural_network.num_weights))
        num_samples = self._X.shape[0]
        num_outputs = self._neural_network.output_shape[0]
        
        for i in range(num_outputs):
            # similar to what is in the objective, but we compute a matrix multiplication of
            # weight probability gradients and a loss vector.
            grad += weight_prob_grad[:, i, :].T @ self._loss(np.full(num_samples, i), self._y)

        grad = grad / self._num_samples
        return grad
    
    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
       
        return ClassifierMixin.score(self, X, y, sample_weight)
    
    def predict(self, X: np.ndarray) -> np.ndarray: 
       
        if self._neural_network.output_shape == (1,):
            predict = np.sign(self._neural_network.forward(X, self._fit_result.x))
       
        else:
            forward = self._neural_network.forward(X, self._fit_result.x)
            predict_ = np.argmax(forward, axis=1)
            predict = predict_
        
        return predict