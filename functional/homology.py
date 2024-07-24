import numpy as np
from gph import ripser_parallel
from gudhi.wasserstein.barycenter import lagrangian_barycenter
from scipy.spatial import distance_matrix
from utils.math import unique_points, inf_mask, extended_distance
from utils.matching import matching_alg

def vr_diagrams(X: np.array, maxdim: int = 1, distances: bool = False, gens: bool = False):
    """
    Computes Vietoris-Rips persistence diagrams.

    Args:
        X : np.array
            Input data, either a point cloud or distance matrix.
        maxdim : int
            Maximum dimension of homology.
        distances : bool
            Whether X is a distance matrix.
        gens : bool
            Whether to return generators.

    Returns:
        list[np.ndarray]:
            Persistence diagrams and optionally generators.
    """
    if not gens:
        return ripser_parallel(X, maxdim=maxdim, metric='precomputed' if distances else 'euclidean')['dgms']
    ret = ripser_parallel(X, maxdim=maxdim, metric='precomputed' if distances else 'euclidean', return_generators=True)
    return ret['dgms'], ret['gens']

def drop_inf(diag: list[np.ndarray]) -> list[np.ndarray]:
    """
    Removes infinite values from persistence diagrams.

    Args:
        diag (list[np.ndarray]):
            Persistence diagrams.

    Returns:
        list[np.ndarray]:
            Cleaned persistence diagrams.
    """
    for dim in range(len(diag)):
        mask = inf_mask(diag[dim])
        if mask.shape:
            diag[dim] = diag[dim][~mask.any(axis=1)].reshape(-1, 2)
    return diag

def diagrams_barycenter(diag: list[list[np.ndarray]]) -> list[np.ndarray]:
    """
    Computes the barycenters of persistence diagrams.

    Args:
        diag (list[list[np.ndarray]]):
            Persistence diagrams.

    Returns:
        list[np.ndarray]:
            Barycenters of persistence diagrams.
    """
    diag = list(map(drop_inf, diag))
    bary = [lagrangian_barycenter([d[dim] for d in diag]) for dim in range(len(diag))]
    return bary

def betti(diag, n_bins: int = 100):
    """
    Computes Betti numbers of persistence diagrams.

    Args:
        diag (list[np.ndarray]):
            Persistence diagrams.
        n_bins (int):
            Number of bins.

    Returns:
        np.ndarray:
            Betti numbers.
    """
    diag = drop_inf(diag)
    global_min = min(d.min() if d.size else 0 for d in diag)
    global_max = max(d.max() if d.size else 0 for d in diag)
    steps = np.linspace(global_min, global_max, num=n_bins, endpoint=True)
    bc = [
        ((d[:, 0] <= steps.reshape(-1, 1)) & (d[:, 1] > steps.reshape(-1, 1))).sum(axis=1) if d.size else np.zeros_like(steps)
        for d in diag
    ]
    return np.array(bc)

def euler(bc: np.ndarray):
    """
    Computes Euler characteristic from Betti curves.

    Parameters:
        bc (np.ndarray):
            Betti curves.

    Returns:
        int:
            Euler characteristic.
    """
    return bc[::2].sum(axis=-2) - bc[1::2].sum(axis=-2)

def persistence_entropy(diag):
    """
    Computes persistence entropy of persistence diagrams.

    Parameters:
        diag (list[np.ndarray]):
            Persistence diagrams.

    Returns:
        np.ndarray:
            Persistence entropy.
    """
    diag = drop_inf(diag)
    L = persistence_norm(diag)
    prob = [(d[:, 1] - d[:, 0]) / L[dim] if d.size else None for dim, d in enumerate(diag)]
    return np.array([entropy(p, np.log) if p is not None else 0 for p in prob])

def persistence_norm(diag: list[np.ndarray]) -> np.ndarray:
    """
    Computes sums of lifetimes for persistence diagrams.

    Parameters:
        diag (list[np.ndarray]):
            Persistence diagrams.

    Returns:
        np.ndarray:
            Sums of lifetimes.
    """
    diag = drop_inf(diag)
    return np.array([np.sum(d[:, 1] - d[:, 0]) for d in diag])

def total_persistence(diag: list[np.ndarray], q: float) -> float:
    """
    Computes total persistence of persistence diagrams.

    Parameters:
        diag (list[np.ndarray]):
            Persistence diagrams.
        q (float):
            Exponent.

    Returns:
        float:
            Total persistence.
    """
    diag = np.vstack(drop_inf(diag))
    return np.power(diag[:, 1] - diag[:, 0], q).sum()

def amplitude(diag: list[np.ndarray], p: float) -> float:
    """
    Computes amplitude of persistence diagrams.

    Parameters:
        diag (list[np.ndarray]):
            Persistence diagrams.
        p (float):
            Exponent.

    Returns:
        float:
            Amplitude.
    """
    diag = np.vstack(drop_inf(diag))
    if p == np.inf:
        return np.max(diag[:, 1] - diag[:, 0]) / np.sqrt(2)
    return np.power(total_persistence([diag], p), 1 / p) / np.sqrt(2)

def landscapes(diag: list[np.ndarray], n_points: int = 100) -> np.ndarray:
    """
    Computes persistence landscapes.

    Parameters:
        diag (list[np.ndarray]):
            Persistence diagrams.
        n_points (int):
            Number of points in the landscape.

    Returns:
        np.ndarray:
            Persistence landscapes.
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
        diag (list[np.ndarray]):
            Persistence diagrams.
        n_points (int):
            Number of points in the landscape.
        p (float):
            Exponent.

    Returns:
        float:
            Landscape norm.
    """
    l = landscapes(drop_inf(diag), n_points)
    return np.linalg.norm(np.linalg.norm(l, p, axis=-1), p)

def pairwise_dist(bc: np.array):
    """
    Computes pairwise distance matrices for Betti curves.

    Parameters:
        bc (np.array):
            Betti curves.

    Returns:
        list[np.ndarray]:
            Distance matrices.
    """
    return [distance_matrix(bc[:, dim], bc[:, dim]) for dim in range(bc.shape[1])]

# def wasserstein_distance(diagX: list[np.ndarray], diagY: list[np.ndarray], q: float = np.inf, matching: bool = False) -> [tuple[float, np.ndarray], float]:
#     """
#     Computes Wasserstein distance between persistence diagrams.

#     Parameters:
#         diagX (list[np.ndarray]):
#             First persistence diagram.
#         diagY (list[np.ndarray]):
#             Second persistence diagram.
#         q (float):
#             Exponent.
#         matching (bool):
#             Whether to return the matching.

#     Returns:
#         tuple[float, np.ndarray], float:
#             Wasserstein distance and optionally the matching.
#     """
#     diagX, diagY = np.vstack(drop_inf(diagX)), np.vstack(drop_inf(diagY))
#     diagXp, diagYp = diagX.mean(axis=1) / 2, diagY.mean(axis=1) / 2
#     mat = matching_alg(extended_distance(diagX, diagY, q))
#     norm = np.linalg.norm(
#         np.max(
#             np.abs(np.vstack([diagX, diagXp])[mat] - np.vstack([diagY, diagYp])), axis=1
#         ), q
#     )
#     if matching:
#         return norm, mat
#     return norm

# def frechet_mean(diag: list[np.ndarray], q: float = np.inf) -> np.ndarray:
#     """
#     Computes Fréchet mean of persistence diagrams.

#     Parameters:
#         diag (list[np.ndarray]):
#             Persistence diagrams.
#         q (float):
#             Exponent.

#     Returns:
#         np.ndarray:
#             Fréchet mean.
#     """
#     diag = drop_inf(diag)
#     Y = diag[0]
#     stop = False
#     while not stop:
#         k = len(Y)
#         y: list[np.ndarray] = [np.empty(0)] * k
#         for i, d in enumerate(diag):
#             diagXp, diagYp = d.mean(axis=1) / 2, Y.mean(axis=1) / 2
#             dist = extended_distance(Y, d, q)
#             mat = matching_alg(dist)[:k]
#             t = np.zeros_like(mat)
#             for j in range(len(mat)):
#                 t[mat[j]] = j
#             y[i] = np.vstack([d, diagYp])[t[:k]].reshape(-1, 1, 2)
#         y = np.hstack(y).mean(axis=-2)
#         if np.array_equal(y, Y):
#             stop = True
#         else:
#             Y = y
#     return Y

# def frechet_variance(diag: list[np.ndarray], Y: np.ndarray = None, q: float = np.inf) -> float:
#     """
#     Computes Fréchet variance of persistence diagrams.

#     Parameters:
#         diag (list[np.ndarray]):
#             Persistence diagrams.
#         Y (np.ndarray, optional):
#             Fréchet mean of the persistence diagrams.
#         q (float):
#             Exponent.

#     Returns:
#         float:
#             Fréchet variance.
#     """
#     Y = Y or frechet_mean(diag, q)
#     return np.sum([np.square(wasserstein_distance([d], [Y], q)) for d in diag]) / (len(diag) - 1)

def cross_barcode(X: np.array, Y: np.array, maxdim: int = 1):
    """
    Computes the cross barcode of two point clouds.

    Parameters:
        X (np.array):
            First point cloud.
        Y (np.array):
            Second point cloud.
        maxdim (int):
            Maximum dimension of the homology group to compute.

    Returns:
        list[np.ndarray]:
            Persistence diagrams for the cross barcode.
    """
    X = unique_points(X)
    Y = unique_points(Y)
    return vr_diagrams(np.vstack([X, Y]), maxdim=maxdim)

def r_cross_barcode(X: np.array, Y: np.array, maxdim: int = 1):
    """
    Computes the r cross barcode of two point clouds using distance matrices.

    Parameters:
        X (np.array):
            First point cloud.
        Y (np.array):
            Second point cloud.
        maxdim (int):
            Maximum dimension of the homology group to compute.

    Returns:
        list[np.ndarray]:
            Persistence diagrams for the r cross barcode.
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
        X (np.array):
            First point cloud.
        Y (np.array):
            Second point cloud.
        maxdim (int):
            Maximum dimension of the homology group to compute.

    Returns:
        np.ndarray:
            MTD scores.
    """
    return persistence_norm(cross_barcode(X, Y, maxdim))

def rtd(X: np.array, Y: np.array, maxdim: int = 1):
    """
    Computes the RTD scores for two point clouds.

    Parameters:
        X (np.array):
            First point cloud.
        Y (np.array):
            Second point cloud.
        maxdim (int):
            Maximum dimension of the homology group to compute.

    Returns:
        np.ndarray:
            RTD scores.
    """
    return persistence_norm(r_cross_barcode(X, Y, maxdim))


# def ls_moment(diag: list[np.ndarray]):
#     z = persistence_norm(diag)
#     return np.sum(z[::2] - z[1::2])
