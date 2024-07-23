import numpy as np
from gph import ripser_parallel
from gudhi.wasserstein.barycenter import lagrangian_barycenter
from scipy.spatial import distance_matrix
from .information import entropy
from utils.math import unique_points, inf_mask, extended_distance
from utils.matching import matching_alg


def vr_diagrams(X: np.array, maxdim: int = 1, distances: bool = False, gens: bool = False):
    """
    Computes Vietoris-Rips persistence diagrams for a given point cloud or distance matrix.

    Parameters:
    X (np.array): Input data, either a point cloud or a distance matrix.
    maxdim (int): Maximum dimension of the homology group to compute.
    distances (bool): Whether X is a distance matrix.
    gens (bool): Whether to return the generators of the persistent homology.

    Returns:
    list[np.ndarray]: Persistence diagrams and optionally the generators.
    """
    if not gens:
        return ripser_parallel(X, maxdim=maxdim, metric='precomputed' if distances else 'euclidean')['dgms']
    ret = ripser_parallel(X, maxdim=maxdim, metric='precomputed' if distances else 'euclidean', return_generators=True)
    return ret['dgms'], ret['gens']


def drop_inf(diag: list[np.ndarray]) -> list[np.ndarray]:
    """
    Removes infinite values from persistence diagrams.

    Parameters:
    diag (list[np.ndarray]): A list of persistence diagram.

    Returns:
    list[np.ndarray]: Cleaned persistence diagram without infinite values.
    """
    for dim in range(len(diag)):
        mask = inf_mask(diag[dim])
        if mask.shape:
            diag[dim] = diag[dim][~mask.any(axis=1)].reshape(-1, 2)
    return diag


def diagrams_barycenter(diag: list[list[np.ndarray]]) -> list[np.ndarray]:
    """
    Computes the barycenters of a list of persistence diagrams.

    Parameters:
    diag (list[list[np.ndarray]]): A list of lists of persistence diagrams.

    Returns:
    list[np.ndarray]: The barycenters of the persistence diagrams.
    """
    diag = list(map(drop_inf, diag))
    bary = []
    for dim in range(len(diag)):
        bary.append(lagrangian_barycenter([d[dim] for d in diag]))
    return bary


def betti(diag, n_bins: int = 100):
    """
    Computes the Betti numbers of the persistence diagrams.

    Parameters:
    diag (list[np.ndarray]): A list of persistence diagram.
    n_bins (int): Number of bins for the Betti curve.

    Returns:
    np.ndarray: An array of Betti numbers.
    """
    diag = drop_inf(diag)
    global_min = min(diag[dim].min() if diag[dim].size else 0 for dim in range(len(diag)))
    global_max = max(diag[dim].max() if diag[dim].size else 0 for dim in range(len(diag)))
    steps = np.linspace(global_min, global_max, num=n_bins, endpoint=True)
    bc = [
        ((diag[dim][:, 0] <= steps.reshape(-1, 1)) & (diag[dim][:, 1] > steps.reshape(-1, 1))).sum(axis=1) if diag[dim].size else np.zeros_like(steps) for dim in range(len(diag))
    ]

    return np.array(bc)


def euler(bc: np.ndarray):
    """
    Computes the Euler characteristic from Betti curves.

    Parameters:
    bc (np.ndarray): Betti curves.

    Returns:
    int: Euler characteristic.
    """
    return bc[::2].sum(axis=-2) - bc[1::2].sum(axis=-2)


def persistence_entropy(diag):
    """
    Computes the persistence entropy of persistence diagrams.

    Parameters:
    diag (list[np.ndarray]): A list of persistence diagram.

    Returns:
    np.ndarray: Persistence entropy.
    """
    diag = drop_inf(diag)
    L = persistence_norm(diag)
    prob = [(diag[dim][:, 1] - diag[dim][:, 0]) / L[dim] if diag[dim].size else None for dim in range(len(diag))]
    return np.array([entropy(prob[dim], np.log) if prob[dim] is not None else 0 for dim in range(len(diag))])


def persistence_norm(diag: list[np.ndarray]) -> np.ndarray:
    """
    Computes the sums of lifetimes for all homology dimensions of persistence diagrams.

    Parameters:
    diag (list[np.ndarray]): A list of persistence diagrams.

    Returns:
    np.ndarray: Sums of lifetimes for each homology dimension.
    """
    diag = drop_inf(diag)
    z = np.zeros(len(diag))
    for dim in range(len(diag)):
        for start, end in diag[dim]:
            z[dim] += end - start
    return z


def total_persistence(diag: list[np.ndarray], q: float) -> float:
    """
    Computes sum of the power-weighted sums of lifetimes for all homology dimensions raised to the power `q`.

    Parameters:
    diag (list[np.ndarray]): A list of persistence diagrams.
    q (float): Exponent.

    Returns:
    float: Total persistence.
    """
    diag = np.vstack(drop_inf(diag))
    return np.power(diag[:, 1] - diag[:, 0], q).sum()


def amplitude(diag: list[np.ndarray], p: float) -> float:
    """
    Computes the amplitude of persistence diagrams.

    Parameters:
    diag (list[np.ndarray]): A list of persistence diagrams.
    p (float): Exponent.

    Returns:
    float: Amplitude.
    """
    diag = np.vstack(drop_inf(diag))
    if p == np.inf:
        return np.max(diag[:, 1] - diag[:, 0]) / np.sqrt(2)
    return np.power(total_persistence([diag], p), 1 / p) / np.sqrt(2)


def landscapes(diag: list[np.ndarray], n_points: int = 100) -> np.ndarray:
    """
    Computes the persistence landscapes of persistence diagrams.

    Parameters:
    diag (list[np.ndarray]): A list of persistence diagrams.
    n_points (int): Number of points in the landscape.

    Returns:
    np.ndarray: Persistence landscapes.
    """
    diag = np.vstack(drop_inf(diag))
    global_min, global_max = diag.min(), diag.max()
    steps = np.linspace(global_min, global_max, num=n_points, endpoint=True)
    ans = np.maximum(np.minimum(steps.reshape(-1, 1) - diag[:, 0], -steps.reshape(-1, 1) + diag[:, 1]), 0)
    return np.sort(ans, axis=-1)[:, ::-1]


def landscape_norm(diag: list[np.ndarray], n_points: int = 100, p: float = np.inf) -> float:
    """
    Computes the norm of persistence landscapes.

    Parameters:
    diag (list[np.ndarray]): A list of persistence diagrams.
    n_points (int): Number of points in the landscape.
    p (float): Exponent.

    Returns:
    float: Landscape norm.
    """
    l = landscapes(drop_inf(diag), n_points)
    return np.linalg.norm(np.linalg.norm(l, p, axis=-1), p)


# def ls_moment(diag: list[np.ndarray]):
#     z = persistence_norm(diag)
#     return np.sum(z[::2] - z[1::2])


def pairwise_dist(bc: np.array):
    """
    Computes pairwise distance matrices for Betti curves.

    Parameters:
    bc (np.array): Betti curves.

    Returns:
    list[np.ndarray]: List of distance matrices for each dimension.
    """
    return [
        distance_matrix(bc[:, dim], bc[:, dim]) for dim in range(bc.shape[1])
    ]


def wasserstein_distance(diagX: list[np.ndarray], diagY: list[np.ndarray], q: float = np.inf, matching: bool = False) -> [tuple[float, np.ndarray], float]:
    """
    Computes the Wasserstein distance between two persistence diagrams.

    Parameters:
    diagX (list[np.ndarray]): First persistence diagram.
    diagY (list[np.ndarray]): Second persistence diagram.
    q (float): Exponent for the distance.
    matching (bool): Whether to return the matching.

    Returns:
    tuple[float, np.ndarray], float: Wasserstein distance and optionally the matching.
    """
    diagX, diagY = np.vstack(drop_inf(diagX)), np.vstack(drop_inf(diagY))
    diagXp, diagYp = diagX.mean(axis=1) / 2, diagY.mean(axis=1) / 2
    mat = matching_alg(extended_distance(diagX, diagY, q))
    norm = np.linalg.norm(
        np.max(
            np.abs(np.vstack([diagX, diagXp])[mat] - np.vstack([diagY, diagYp])), axis=1
        ), q
    )
    if matching:
        return norm, mat
    return norm


def frechet_mean(diag: list[np.ndarray], q: float = np.inf) -> np.ndarray:
    """
    Computes the Fréchet mean of persistence diagrams.

    Parameters:
    diag (list[np.ndarray]): A list of persistence diagrams.
    q (float): Exponent for the distance.

    Returns:
    np.ndarray: Fréchet mean of the persistence diagrams.
    """
    diag = drop_inf(diag)
    Y = diag[0]
    stop = False
    while not stop:
        k = len(Y)
        y: list[np.ndarray] = [np.empty(0)] * k
        for i, d in enumerate(diag):
            diagXp, diagYp = d.mean(axis=1) / 2, Y.mean(axis=1) / 2
            dist = extended_distance(Y, d, q)
            # mat = _matching_alg(dist)[:k]
            mat = matching_alg(dist)[:k]
            t = np.zeros_like(mat)
            for j in range(len(mat)):
                t[mat[j]] = j
            y[i] = np.vstack([d, diagYp])[t[:k]].reshape(-1, 1, 2)  # mapping for Y (off-diagonal)
        y = np.hstack(y).mean(axis=-2)
        if y == Y:
            stop = True
        else:
            Y = y
    return Y


def frechet_variance(diag: list[np.ndarray], Y: np.ndarray = None, q: float = np.inf) -> float:
    """
    Computes the Fréchet variance of persistence diagrams.

    Parameters:
    diag (list[np.ndarray]): A list of persistence diagrams.
    Y (np.ndarray, optional): Fréchet mean of the persistence diagrams.
    q (float): Exponent for the distance.

    Returns:
    float: Fréchet variance of the persistence diagrams.
    """
    Y = Y or frechet_mean(diag, q)
    return np.sum([np.square(wasserstein_distance([d], [Y], q)) for d in diag]) / (len(diag) - 1)


def cross_barcode(X: np.array, Y: np.array, maxdim: int = 1):
    """
    Computes the cross barcode of two point clouds.

    Parameters:
    X (np.array): First point cloud.
    Y (np.array): Second point cloud.
    maxdim (int): Maximum dimension of the homology group to compute.

    Returns:
    list[np.ndarray]: Persistence diagrams for the cross barcode.
    """
    X = unique_points(X)
    Y = unique_points(Y)
    return vr_diagrams(np.vstack([X, Y]), maxdim=maxdim)


def r_cross_barcode(X: np.array, Y: np.array, maxdim: int = 1):
    """
    Computes the relaxed cross barcode of two point clouds using distance matrices.

    Parameters:
    X (np.array): First point cloud.
    Y (np.array): Second point cloud.
    maxdim (int): Maximum dimension of the homology group to compute.

    Returns:
    list[np.ndarray]: Persistence diagrams for the relaxed cross barcode.
    """
    X = unique_points(X)
    Y = unique_points(Y)
    XX = distance_matrix(X, X)
    YY = distance_matrix(Y, Y)
    inf_block = np.triu(np.full_like(XX, np.inf), 1) + XX

    M = np.block([
        [XX, inf_block.T, np.zeros((XX.shape[0], 1))],
        [inf_block, np.minimum(XX, YY), np.full((XX.shape[0], 1), np.inf)],
        [np.zeros((1, XX.shape[0])), np.full((1, XX.shape[0]), np.inf), 0]
    ])
    return vr_diagrams(M, maxdim=maxdim, distances=True)


def mtd(X: np.array, Y: np.array, maxdim: int = 1):
    """
    Computes the MTD scores for two point clouds.

    Parameters:
    X (np.array): First point cloud.
    Y (np.array): Second point cloud.
    maxdim (int): Maximum dimension of the homology group to compute.

    Returns:
    np.ndarray: MTD scores.
    """
    return persistence_norm(cross_barcode(X, Y, maxdim))


def rtd(X: np.array, Y: np.array, maxdim: int = 1):
    """
    Computes the RTD scores for two point clouds.

    Parameters:
    X (np.array): First point cloud.
    Y (np.array): Second point cloud.
    maxdim (int): Maximum dimension of the homology group to compute.

    Returns:
    np.ndarray: RTD scores.
    """
    return persistence_norm(r_cross_barcode(X, Y, maxdim))
