import pickle
import gzip
import numpy as np
import os

from base_network import BaseNetwork
from quantum_network import QuantumNetwork
from quantuminspire.credentials import save_account
from quantuminspire.credentials import get_authentication
from quantuminspire.qiskit import QI

from qiskit import execute
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit

def initialize_QI():
    
    os.environ['QI_TOKEN'] = 'eac3d1bca7ee7d56743cefdc238e9f7abdaf15e9'    
    QI_URL = os.getenv('API_URL', 'https://api.quantum-inspire.com/')

    project_name = 'Qiskit-entangle'
    authentication = get_authentication()
    
    QI.set_authentication(authentication, QI_URL, project_name=project_name)
    qi_backend = QI.get_backend('QX single-node simulator')
    
    q = QuantumRegister(2)
    b = ClassicalRegister(2)
    circuit = QuantumCircuit(q, b)

    circuit.h(q[0])
    circuit.cx(q[0], q[1])
    circuit.measure(q, b)

    qi_job = execute(circuit, backend=qi_backend, shots=256)
    qi_result = qi_job.result()
    histogram = qi_result.get_counts(circuit)
    
    print(histogram)


def run_base_network():

    training_data, validation_data, test_data = load_data_together()
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
 
def load_data_together():
    train, validate, test = load_data()
    train_x = [np.reshape(x, (784, 1)) for x in train[0]]
    train_y = [one_hot_encode(y) for y in train[1]]
    training_data = zip(train_x, train_y)
    validate_x = [np.reshape(x, (784, 1)) for x in validate[0]]
    validate_y = [one_hot_encode(y) for y in validate[1]]
    validation_data = zip(validate_x, validate_y)
    test_x = [np.reshape(x, (784, 1)) for x in test[0]]
    test_y = [one_hot_encode(y) for y in test[1]]
    testing_data = zip(test_x, test_y)
    return (training_data, validation_data, testing_data)

def main():
    
    initialize_QI()
    #run_base_network()


if __name__ == '__main__':
    main()
    