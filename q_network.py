from qiskit_machine_learning.neural_networks.sampling_neural_network import SamplingNeuralNetwork
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import Gradient, CircuitSampler, StateFn, OpflowError, OperatorBase
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance

from typing import List, Union, Tuple, Callable

class QNetwork(SamplingNeuralNetwork):
    
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
        
        self._input_params = input_params
        self._weight_params = weight_params
        self._input_gradients = input_gradients

        self._circuit = circuit.copy()
        self._circuit_transpiled = False
       
        self._original_output_shape = output_shape
        self._output_shape = output_shape
        
        self._interpret = interpret
        self._quantum_instance = quantum_instance
        

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
            self._input_params + self._weight_params)
        

   