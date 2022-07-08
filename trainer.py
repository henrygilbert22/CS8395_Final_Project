import numpy as np
import matplotlib.pyplot as plt
import pickle

from quantum_network import QuantumNetwork
from base_network import BaseNetwork

def load_data_batch(file_name: str):

    with open(file_name, 'rb') as f:

        datadict = pickle.load(f)
        
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        
        return X, Y

def load_CIFAR10():
    """ load all of cifar """
    
    xs = []
    ys = []
    
    for num in range(1,6):
        X, Y = load_data_batch(f"data/data_{num}.pkl")
        xs.append(X)
        ys.append(Y)    

    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    Xte, Yte = load_data_batch(f"test.pkl")
    
    return Xtr, Ytr, Xte, Yte


def process_CIFAR10_data(num_training=49000, num_validation=1000, 
    num_test=1000) -> tuple([np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """ Load the CIFAR-10 dataset from disk and perform preprocessing

    Args:
        num_training: number of training examples
        num_validation: number of validation examples
        num_test: number of test examples  
    
    Returns:
        X_train: training data matrix
        y_train: training label matrix
        X_val: validation data matrix
    """
    # Load the raw CIFAR-10 data
    
    X_train, y_train, X_test, y_test = load_CIFAR10()
        
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test

def train_base_network():

    X_train, y_train, X_val, y_val, X_test, y_test = process_CIFAR10_data()
    net = BaseNetwork(3072, 100, 10)

    training_metrics = net.train(X_train, y_train, X_val, y_val,
                            num_iters=100, batch_size=200,
                            learning_rate=1e-4, learning_rate_decay=0.95,
                            reg=0.25)

    create_training_graphs(training_metrics)


def create_training_graphs(training_metrics: dict):

    plt.subplot(2, 1, 1)
    plt.plot(training_metrics['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(training_metrics['train_acc_history'], label='train')
    plt.plot(training_metrics['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.legend()
    plt.savefig('graphs/base_training_graph.png')


def main():
    
    train_base_network()
    # train_quantum_network()

if __name__ == '__main__':
    main()