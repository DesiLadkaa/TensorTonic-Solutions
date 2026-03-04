import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    #Write code here
    x = np.array(x, dtype=float)  # Ensure input is a NumPy array of floats
    # Numerically stable vectorized computation
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )