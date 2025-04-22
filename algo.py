import numpy as np
from scipy.linalg import eigh, pinv
from scipy.sparse.csgraph import shortest_path
from typing import Tuple

# --- Step 1: Compute Degree Matrix ---
def get_degree(adj:np.array) -> np.array:
    """
    Finds the degree of each node in the matrix
    Will avoid any zeros instead setting it to e^-10.

    Args:
        adj (np.array): matrix

    Returns:
        np.array: degree matrix
    """
    return np.diag(np.sum(adj, axis=1))

# --- Step 2: Normalized Laplacian ---
def find_normalized_laplacian(adj:np.array, degree:np.array) -> np.array:
    """
    Compute the normalized graph Laplacian matrix.

    Args:
        adj (np.array): _description_
        degree (np.array): _description_

    Returns:
        np.array: normalized laplacian matrix
    """
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree)))
    laplacian = np.eye(len(adj)) - d_inv_sqrt @ adj @ d_inv_sqrt
    return laplacian

# --- Step 3: Sorted Eigenvalues & Vectors ---
def get_sorted_eigen(laplacian:np.array) -> Tuple[np.array, np.array]:
    eigenvalues, eigenvectors = eigh(laplacian)
    return eigenvalues, eigenvectors

# --- Step 4: Estimate Cluster Number ---
def findClusterAmount(eigenvalues:np.array) -> int:
    gaps = np.diff(eigenvalues)
    return int(np.argmax(gaps) + 1)

# --- Step 5: Normalize Eigenvector Rows ---
def normalize_rows(mat:np.array) -> np.array:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / norms

# --- Step 6: K-Means Until Convergence ---
def kmeans_until_converge(data:np.array, k:int, max_iters=100) -> list[list[int]]:
    n, d = data.shape
    np.random.seed(0)
    centers = data[np.random.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iters):
        distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.all(labels == new_labels):
            break
        labels = new_labels
        for j in range(k):
            cluster_points = data[labels == j]
            if len(cluster_points) > 0:
                centers[j] = np.mean(cluster_points, axis=0)

    clusters = [[] for _ in range(k)]
    for i, label in enumerate(labels):
        clusters[label].append(i + 1)
    return clusters

# --- Step 7: Rewire Edges to Boost Connectivity ---
def rewire_for_connectivity(adj:np.array, clusters:list[list[int]], num_edges_to_add=3) -> np.array:
    n = adj.shape[0]
    new_adj = adj.copy()
    cluster_map = {}
    for idx, cluster in enumerate(clusters):
        for node in cluster:
            cluster_map[node] = idx

    inter_cluster_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if cluster_map[i + 1] != cluster_map[j + 1] and adj[i, j] == 0:
                inter_cluster_pairs.append((i, j))

    dist_matrix = shortest_path(adj, directed=False, unweighted=True)
    scored_pairs = [(i, j, dist_matrix[i, j]) for (i, j) in inter_cluster_pairs if not np.isinf(dist_matrix[i, j])]
    scored_pairs.sort(key=lambda x: -x[2])

    edges_added = 0
    for (i, j, _) in scored_pairs:
        if edges_added >= num_edges_to_add:
            break
        new_adj[i, j] = 1
        new_adj[j, i] = 1
        edges_added += 1

    return new_adj

# --- Step 8: Full Iterative Pipeline Until Convergence ---
def compute_connectivity_until_convergence(matrix, max_iters=100, edges_per_iter=2) -> Tuple[list[list[int]], int, np.array]:
    adj = matrix.copy()
    previous_adj = None
    iteration = 0

    while iteration < max_iters and (previous_adj is None or not np.array_equal(adj, previous_adj)):
        previous_adj = adj.copy()
        degree = get_degree(adj)
        laplacian = find_normalized_laplacian(adj, degree)
        eigenvalues, eigenvectors = get_sorted_eigen(laplacian)
        num_clusters = findClusterAmount(eigenvalues) + 1
        norm_eigvecs = normalize_rows(eigenvectors[:, :num_clusters])
        clusters = kmeans_until_converge(norm_eigvecs, num_clusters)
        adj = rewire_for_connectivity(adj, clusters, num_edges_to_add=edges_per_iter)
        iteration += 1

    return clusters, num_clusters, adj

# Example usage with 6-node graph
if __name__ == "__main__":
    A = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ])

    clusters, k, new_adj = compute_connectivity_until_convergence(A)
    print("Final clusters:", clusters)
    print("New adjacency matrix:\n", new_adj)
