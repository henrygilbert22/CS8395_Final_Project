import numpy as np
from typing import Tuple

from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.algorithms.optimizers import COBYLA

from base_network import BaseNetwork
from quantum_classifier import QuantumClassifier
from quantum_network import QuantumNeuralNetwork

def run_base_network() -> None:
    """ Runs the base network on the given
        data and prints the classification score.
        
    Args:
        None
    
    Returns:
        None
    """

    training_data = load_base_data()
    testing_data = load_base_data()

    net = BaseNetwork([2, 30, 30, 2])
    net.learn(training_data, 10, 10, 3.0, test_data=testing_data)

def hot_encode(y: list, num_classes: int) -> np.array:
    """ Converts a list of labels to a one-hot encoding.
    
    Args:
        y: The list of labels.
        num_classes: The number of classes.
    
    Returns:
        The one-hot encoding.
    """

    encoded = np.zeros((num_classes, 1))
    encoded[y] = 1.0
    return encoded

def load_base_data() -> list:
    """ Loads a toy dataset for the base network. A bit 
        redundant in relation to load_q_data, the main
        difference being the labeles needed to be encoded.
        
    Args:
        None
    
    Returns:
        The training data.
    """
    
    num_inputs = 2
    num_samples = 20
    num_outputs = 2
    
    X = 2 * algorithm_globals.random.random([num_samples, num_inputs]) - 1
    y01 = 1 * (np.sum(X, axis=1) >= 0)  # in { 0,  1}
    
    X = [np.reshape(x, (2, 1)) for x in X]
    Y = [hot_encode(y, num_outputs) for y in y01]

    return zip(X, Y)

def load_q_data() -> Tuple[np.array, np.array]:
    """ Loads a toy dataset for the quantum network.
    
    Args:
        None
    
    Returns:
        The training data.
    """
    
    num_inputs = 2
    num_samples = 20
    
    X = 2 * algorithm_globals.random.random([num_samples, num_inputs]) - 1
    y01 = 1 * (np.sum(X, axis=1) >= 0)  # in { 0,  1}
    
    return X, y01
    
def run_quantum_network() -> None:
    """ Runs the quantum network on the given
        data and prints the classification score.
        
    Args:
        None
    
    Returns:
        None
    """
    
    num_inputs = 2
    num_outputs = 2
    
    train_X, train_y = load_q_data()
    
    # Initializing the quantum instance and setting the backend to Aer.
    quantum_instance = QuantumInstance(Aer.get_backend("aer_simulator"), shots=1024)
    feature_map = ZZFeatureMap(num_inputs)
    ansatz = RealAmplitudes(num_inputs, reps=3)
    
    # Creating a quantum circuit and adding the feature map with ansatz.
    qc = QuantumCircuit(num_inputs)
    qc.append(feature_map, range(num_inputs))
    qc.append(ansatz, range(num_inputs))

    # Defining the interpreter function for the quantum network.
    def parity_func(x):
        return "{:b}".format(x).count("1") % 2
    
    circuit_qnn = QuantumNeuralNetwork(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        interpret=parity_func,
        output_shape=num_outputs,
        quantum_instance=quantum_instance,
    )
    
    circuit_classifier = QuantumClassifier(neural_network=circuit_qnn, optimizer=COBYLA())
    circuit_classifier.learn(train_X, train_y)
    
    score = circuit_classifier.score(train_X, train_y)
    print(f"Quantum Neural Network Classification Score: {score}")
    
def main():
    
    run_base_network()
    run_quantum_network()


if __name__ == '__main__':
    main()
    