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
    
    _circuit: QuantumCircuit
    
    _original_output_shape: Union[int, Tuple[int, ...]]
    _interpret: Callable[[int], Union[int, Tuple[int, ...]]]
    _quantum_instance: QuantumInstance
    
    def __init__(
        self,
        circuit: QuantumCircuit,
        input_params: List[Parameter],
        weight_params: List[Parameter],
        interpret: Callable[[int], Union[int, Tuple[int, ...]]],
        output_shape: Union[int, Tuple[int, ...]],
        gradient: Gradient,
        quantum_instance: QuantumInstance,
        input_gradients: bool = False,
        ) -> None:
        
        self._input_params = input_params
        self._weight_params = weight_params
        self._input_gradients = input_gradients

        self._circuit = circuit.copy()
        self._circuit_transpiled = False
       
        self._original_output_shape = output_shape

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
        self._gradient = gradient or Gradient()

        self._construct_gradient_circuit()
    
    def _set_quantum_instance(self) -> None:
       

        if self._quantum_instance is not None:
            # add measurements in case none are given
            if self._quantum_instance.is_statevector:
                if len(self._circuit.clbits) > 0:
                    self._circuit.remove_final_measurements()
            elif len(self._circuit.clbits) == 0:
                self._circuit.measure_all()

            # set interpret and compute output shape
            self.set_interpret(interpret, output_shape)

            # prepare sampler
            self._sampler = CircuitSampler(self._quantum_instance, param_qobj=False, caching="all")

            # transpile the QNN circuit
            try:
                self._circuit = self._quantum_instance.transpile(
                    self._circuit, pass_manager=self._quantum_instance.unbound_pass_manager
                )[0]
                self._circuit_transpiled = True
            except QiskitError:
                # likely it is caused by RawFeatureVector, we just ignore this error and
                # transpile circuits when it is required.
                self._circuit_transpiled = False
        else:
            self._output_shape = output_shape