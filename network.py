import numpy as np
from abc import ABC, abstractmethod
import random
from typing import Tuple

class Network(ABC):

    num_layers: int
    sizes: list

    biases: list
    weights: list

    def __init__(self, sizes: list) -> None:
        """ The list ``sizes`` contains the number of neurons in the
            respective layers of the network.  For example, if the
            list was [2, 3, 1] then it would be a three-layer network,
            with the first layer containing 2 neurons, the second layer
            3 neurons, and the third layer 1 neuron.  The biases and weights
            for the network are initialized randomly, using a Gaussian
            distribution with mean 0, and variance 1.
            
        Args:
            sizes: A list of integers specifying network architecture.
            
        Returns:
            None
        """

        self.num_layers = len(sizes)
        self.sizes = sizes

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]

    @abstractmethod
    def sigmoid(self, z: int) -> float:
        """ Abstract sigmoid function to be implemented by subclasses.
        
        Args:
            z: The input to the sigmoid function.
        
        Returns:
            The output of the sigmoid function.
        """

        pass
    
    @abstractmethod
    def sigmoid_prime(self, z: int) -> float:
        """ Abstract sigmoid prime function to be implemented by subclasses.
        
        Args:
            z: The input to the sigmoid prime function.
        
        Returns:
            The output of the sigmoid prime function.
        """

        pass

    @abstractmethod
    def backprop(self, x: np.array, y: np.array) -> Tuple[np.array, np.array]:
        """ Abstract backprop function to be implemented by subclasses.
        
        Args:
            x: The input to the network.
            y: The desired output.
            
        Returns:
            The tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function.
        """
        
        pass    

    def feedforward(self, a: np.array) -> np.array:
        """ Compute the feedforward activation of a network.
        
        Args:
            a: The input to the network.
            
        Returns:
            The output of the network.
        """
       
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        
        return a

    def learn(self, training_data: list, epochs: int, mini_batch_size: int, eta: float, test_data) -> None:
        """ Train the neural network using mini-batch stochastic gradient descent.
        
        Args:
            training_data: A list of tuples ``(x, y)`` representing the training inputs and the desired outputs.
            epochs: Number of epochs to train over.
            mini_batch_size: Size of each mini batch.
            eta: Learning rate.
            test_data: A list of tuples ``(x, y)`` representing the test inputs and the desired outputs.
            
        Returns:
            None
        """

        training_data = list(training_data)
        test_data = list(test_data)

        samples = len(training_data)
        n_test = len(test_data)
    
        for j in range(epochs):
            
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, samples, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            print(f"Epoch {j} - Classification Score: {round(100 * (self.evaluate(test_data) / n_test), 3)}")
                
    def cost_derivative(self, output_activations: np.array, y: np.array) -> np.array:
        """ Return the vector of partial derivatives \partial C_x / \partial a for the output activations.
        
        Args:
            output_activations: The output activations.
            y: The desired outputs.
            
        Returns:
            The vector of partial derivatives \partial C_x / \partial a.
        """

        return(output_activations - y)
    
    def update_mini_batch(self, mini_batch: list, eta: float) -> None:
        """ Update the network's weights and biases by applying gradient descent 
            using backpropagation to a single mini batch.
        
        Args:
            mini_batch: The mini batch to update the network with.
            eta: The learning rate.
        
        Returns:
            None
        """
        
        # Initialize the gradient matrices.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Update the gradient matrices.
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Update the weights
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]

        # Update the biases
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
       
    def evaluate(self, test_data: list) -> float:
        """ Return the number of test inputs for which the neural network outputs the correct result.
        
        Args:
            test_data: A list of tuples ``(x, y)`` representing the test inputs and the desired outputs.
        
        Returns:
            The number of test inputs for which the neural network outputs the correct result.
        """
        
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        
        return sum(int(y[x]) for (x, y) in test_results)
