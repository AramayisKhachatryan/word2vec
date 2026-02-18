import numpy as np
from typing import Union

def sigmoid(x: float) -> Union[float, np.ndarray]:
    """  
    Functions for sigmoid. It clips input into -500 and 500 to avoid overflow in np.exp
    
    Args:
        x: Value to calculate sigmoid of
    
    Returns:
        sigmoid of input number
    """
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Function to calculate cosine similarity of 2 given vectors.
    similarity = (v1, v2)/(||v1||*||v2||)

    Args:
        v1: Vector 1
        v2: Vector 2
    
    Returns:
        Cosine similarity between v1 and v2
    """

    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)