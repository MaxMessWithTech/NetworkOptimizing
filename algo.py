# from scipy.linalg import fractional_matrix_power
import numpy as np
from typing import Tuple


def findClusterAmount(eigenValues) -> int:
    """
    _summary_

    Args:
        eigenValues (_type_): _description_

    Returns:
        int: _description_
    """
    maxDif = 0
    clusters = 0
    for i in range(len(eigenValues)-1):
        dif = abs(eigenValues[i]-eigenValues[i+1])
        if( dif > maxDif):
            maxDif = dif
            clusters = i
    return clusters


def kmeans_until_converge(data:np.array, numClusters:int, tol=1e-6, max_iters = 10000) -> list[list[int]]:
    """
    _summary_

    Args:
        data (np.array): _description_
        numClusters (int): _description_
        tol (_type_, optional): _description_. Defaults to 1e-6.
        max_iters (int, optional): _description_. Defaults to 10000.

    Returns:
        list[list[int]]: _description_
    """

    n, dim = data.shape

    rng = np.random.default_rng()
    centroids = data[rng.choice(n, size = numClusters, replace = False)]

    for _ in range(max_iters):
        distances = np.sqrt((data[:, np.newaxis,:] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids  =np.array([
            data[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(numClusters)
        ])

        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids
    
    # Create a list of length numClusters, with each entry being a blank list
    clusters = [[] for _ in range(numClusters)]

    for i, label in enumerate(labels):
        clusters[label].append(i + 1)
    
    return clusters


def normalize_rows(matrix: np.array) -> np.array:
    """
    _summary_

    Args:
        matrix (np.array): _description_

    Returns:
        _type_: _description_
    """
    norms = np.linalg.norm(matrix, axis = 1, keepdims = True)
    return matrix / np.where(norms == 0, 1, norms)


def get_degree(a: np.array) -> np.array:
    """
    Finds the degree of each node in the matrix
    Will avoid any zeros instead setting it to e^-10.

    Args:
        a (np.array): matrix

    Returns:
        np.array: degree matrix
    """

    degree = np.sum(a, axis=1)
    degree[degree == 0] = 1e-10  # Prevent division by zero
    return degree


def find_normalized_laplacian(a: np.array, degree: np.array) -> np.array:
    """
    Compute the normalized graph Laplacian matrix.

    Args:
        a (np.array): _description_
        degree (np.array): _description_

    Returns:
        np.array: normalized laplacian matrix
    """

    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    laplacian =  np.eye(len(a)) - D_inv_sqrt @ a @ D_inv_sqrt
    return laplacian

# DESCRIPTION OF WHATEVER THIS DOES
def get_sorted_eigen(laplacian: np.array) -> Tuple[np.array, np.array]:
    """Compute and sort eigenvalues and eigenvectors of the Laplacian."""
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


# MAIN FUNCTION TO BE CALLED FROM MAIN
def compute_connectivity(matrix: np.array) -> Tuple[list[list[int]], int]:
    """
    MAIN FUNCTION TO BE CALLED FROM MAIN

    Args:
        matrix (np.array): 

    Returns:
        Tuple[list, int]: _description_
    """

    degree = get_degree(matrix)

    laplacian = find_normalized_laplacian(matrix, degree)
    eigenvalues, eigenvectors = get_sorted_eigen(laplacian)

    # Determine number of clusters from eigen gap
    num_clusters = findClusterAmount(eigenvalues) + 1

    # Normalize eigenvectors row-wise
    norm_eigenvectors = normalize_rows(eigenvectors[:, :num_clusters])

    # Run k-means
    clusters = kmeans_until_converge(norm_eigenvectors, num_clusters)

    return clusters, num_clusters