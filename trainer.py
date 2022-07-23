import pickle
import gzip
import numpy as np
from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

from base_network import BaseNetwork
from q_network import QNN, QNNClassifier

def run_base_network():

    training_data, validation_data, test_data = load_all_together(hot_encode=True)
    net = BaseNetwork([784, 30, 30, 10])
    net.SGD(training_data, 10, 10, 3.0, test_data=test_data)

def load_data():
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return (training_data, validation_data, test_data)
 
def one_hot_encode(y):
    encoded = np.zeros((10, 1))
    encoded[y] = 1.0
    return encoded
 
def load_all_together(hot_encode: bool):
    
    train, validate, test = load_data()
    
    train_x = [np.reshape(x, (784, 1)) for x in train[0]]
    train_y = [one_hot_encode(y) if hot_encode else y for y in train[1]]
    training_data = zip(train_x, train_y)
    
    validate_x = [np.reshape(x, (784, 1)) for x in validate[0]]
    validate_y = [one_hot_encode(y) if hot_encode else y for y in validate[1]]
    validation_data = zip(validate_x, validate_y)
    
    test_x = [np.reshape(x, (784, 1)) for x in test[0]]
    test_y = [one_hot_encode(y) if hot_encode else y for y in test[1]]
    testing_data = zip(test_x, test_y)
    
    return (training_data, validation_data, testing_data)

def load_q_data():
    
    num_inputs = 2
    num_samples = 20
    
    X = 2 * algorithm_globals.random.random([num_samples, num_inputs]) - 1
    y01 = 1 * (np.sum(X, axis=1) >= 0)  # in { 0,  1}
    
    return X, y01
    
def run_quantum_network():
    
    num_inputs = 2
    num_outputs = 2
    
    train_X, train_y = load_q_data()
    
    quantum_instance = QuantumInstance(Aer.get_backend("aer_simulator"), shots=1024)
    feature_map = ZZFeatureMap(num_inputs)
    ansatz = RealAmplitudes(num_inputs, reps=1)
    
    qc = QuantumCircuit(num_inputs)
    qc.append(feature_map, range(num_inputs))
    qc.append(ansatz, range(num_inputs))

    def parity(x):
        return "{:b}".format(x).count("1") % 2
    
    circuit_qnn = QNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        interpret=parity,
        output_shape=num_outputs,
        quantum_instance=quantum_instance,
    )
    
    
    circuit_classifier = QNNClassifier(neural_network=circuit_qnn, optimizer=COBYLA())
    circuit_classifier.learn(train_X, train_y)
    score = circuit_classifier.score(train_X, train_y)
    print("Classification score: {}".format(score))
    
def main():
    
    #run_base_network()
    run_quantum_network()


if __name__ == '__main__':
    main()
    