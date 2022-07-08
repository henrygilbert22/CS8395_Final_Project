
def ReLU(x: list) -> list:
    """ ReLU activation function 
    
    Args:
        x: list of numbers
    
    Returns:
        list of numbers
    """

    return x * (x > 0)