import numpy as np
from scipy.spatial import distance_matrix


def magnitude(X: np.ndarray, distances: bool = False):
    """
    Computes the magnitude of a distance matrix or the matrix itself.

    Parameters:
    X (np.ndarray): Input matrix. If `distances` is False, X is assumed to be a point cloud and distances are computed.
    distances (bool): Whether X is already a distance matrix. If False, distances will be computed.

    Returns:
    float: The magnitude, which is the sum of the inverse of the exponentiated distance matrix.
    """
    d = X if distances else distance_matrix(X, X)
    d = np.exp(-d)
    return np.linalg.inv(d).sum()
