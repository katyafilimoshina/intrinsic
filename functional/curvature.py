import numpy as np
from sklearn.neighbors import NearestNeighbors

from .dimension import mle
from .kernel import gaussian_kernel_one
from utils.math import unit_ball_volume, mle_aggregate


def density(X: np.ndarray, k: int, dim: float, distances: bool = False) -> np.ndarray:
    """
    Computes the density estimation of the point cloud using k-nearest neighbors.

    Parameters:
        X (np.ndarray): 
            Input point cloud with shape (n_samples, n_features) or a distance matrix if `distances` is True.
        k (int): 
            Number of nearest neighbors to consider.
        dim (float): 
            Estimated intrinsic dimension of the data.
        distances (bool): 
            Whether X is already a distance matrix. If False, distances are computed.

    Returns:
        np.ndarray: 
            Density estimates for each point in the point cloud.
    """
    nn = NearestNeighbors(n_neighbors=k, metric='precomputed' if distances else 'minkowski').fit(X)
    dist, _ = nn.kneighbors()
    h = np.power(np.arange(1, k + 1), -1 / (dim + 4))
    return np.sum(1 / np.power(h, k) / k * gaussian_kernel_one(dist / h, dim, 1) / k, axis=-1)


def mean_density(X: np.ndarray, k: int, dim: float, distances: bool = False) -> np.ndarray:
    """
    Computes the mean density of the point cloud.

    Args:
        X (np.ndarray): 
            Input point cloud with shape (n_samples, n_features) or a distance matrix 
            if `distances` is True.
        k (int): 
            Number of nearest neighbors to consider.
        dim (float): 
            Estimated intrinsic dimension of the data.
        distances (bool): 
            Whether `X` is already a distance matrix. If False, distances are computed.

    Returns:
        np.ndarray: 
            Mean density estimates for each point in the point cloud.
    """
    nn = NearestNeighbors(n_neighbors=k, metric='precomputed' if distances else 'minkowski').fit(X)
    dist, ix = nn.kneighbors()
    rho = density(X, k, dim, distances)
    return np.arange(1, k + 1) / np.cumsum(1 / rho[ix], axis=-1)


def volume(X: np.ndarray, k: int, dim: float, distances: bool = False) -> np.ndarray:
    """
    Computes the volume estimation of the point cloud.

    Args:
        X (np.ndarray): 
            Input point cloud with shape (n_samples, n_features) or a distance matrix 
            if `distances` is True.
        k (int): 
            Number of nearest neighbors to consider.
        dim (float): 
            Estimated intrinsic dimension of the data.
        distances (bool): 
            Whether `X` is already a distance matrix. If False, distances are computed.

    Returns:
        np.ndarray: 
            Volume estimates for each point in the point cloud.
    """
    mu = mean_density(X, k, dim, distances)
    return np.arange(1, k + 1) / mu / (k - 1)


def quadratic_fit(X: np.ndarray, k: int, dim: float, distances: bool = False) -> np.ndarray:
    """
    Computes a quadratic fit for the point cloud distances.

    Args:
        X (np.ndarray): 
            Input point cloud with shape (n_samples, n_features) or a distance matrix 
            if `distances` is True.
        k (int): 
            Number of nearest neighbors to consider.
        dim (float): 
            Estimated intrinsic dimension of the data.
        distances (bool): 
            Whether `X` is already a distance matrix. If False, distances are computed.

    Returns:
        np.ndarray: 
            Quadratic fit values for each point in the point cloud.
    """
    nn = NearestNeighbors(n_neighbors=k, metric='precomputed' if distances else 'minkowski').fit(X)
    dist, _ = nn.kneighbors()
    vol = volume(X, k, dim, distances)
    y = vol / unit_ball_volume(dim) / np.power(dist, dim)

    return 5 * np.sum(np.square(dist) * (y - 1) * np.diff(dist, prepend=0), axis=-1) / (dist[:, -1] ** 5 - dist[:, 0] ** 5)


def curvature(X: np.ndarray, k: int, dim: float = None, distances: bool = False) -> np.ndarray:
    """
    Computes the curvature of the point cloud.

    Args:
        X (np.ndarray): 
            Input point cloud with shape (n_samples, n_features) or a distance matrix 
            if `distances` is True.
        k (int): 
            Number of nearest neighbors to consider.
        dim (float, optional): 
            Estimated intrinsic dimension of the data. If None, it is computed using `mle_aggregate`.
        distances (bool): 
            Whether `X` is already a distance matrix. If False, distances are computed.

    Returns:
        np.ndarray: 
            Curvature estimates for each point in the point cloud.
    """
    dim = dim or mle_aggregate(mle(X, k, distances))
    return -6 * (dim + 2) * quadratic_fit(X, k, dim, distances)
