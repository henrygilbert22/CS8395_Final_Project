import numpy as np

def ReLU(X: list) -> list:
    """ ReLU activation function 
    
    Args:
        x: list of numbers
    
    Returns:
        list of numbers
    """

    return X * (X > 0)

def softmax_activation(X: list) -> list:
    """ Applies Softmax activation function to
    list of floats 
    
    Args:
        x: list of numbers
    
    Returns:
        list of numbers
    """

    probabilities = [np.exp(x) / np.sum(np.exp(x)) for x in X]
    return np.array(probabilities)