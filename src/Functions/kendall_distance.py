import numpy as np
import statistics
from nptyping import Float, NDArray, Shape

def normalised_kendall_tau_distance(values1: list, values2: list):
    """Compute the Kendall tau distance.
    From https://en.wikipedia.org/wiki/Kendall_tau_distance"""
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (n * (n - 1))


def kendall_matrix_distance(values1: NDArray[Shape['StateDimension, ActionDimension'], Float], values2: NDArray[Shape['StateDimension, ActionDimension'], Float]) -> float:
    """Return the mean of the normalized tau distance"""
    assert values1.shape == values2.shape, "Both matrix must have the same shape"
    num_rows, num_cols = values1.shape
    vec_dist = np.empty(num_rows, dtype = float)
    for row_index in range(num_rows):
        vec_dist[row_index] = normalised_kendall_tau_distance(values1[row_index,], values2[row_index,])
    return statistics.mean(vec_dist)    
