from scipy.linalg import fractional_matrix_power
import numpy as np
from node import Node
from connection import Connection
import array
from sklearn.cluster import KMeans

from create import visualize_graph, make_random_graph, convert_to_np_matrix


def findClusterAmount(eigenValues) -> int:
    maxDif = 0
    clusters = 0
    for i in range(len(eigenValues)-1):
        dif = abs(eigenValues[i]-eigenValues[i+1])
        if( dif > maxDif):
            maxDif = dif
            clusters = i
    return clusters

def kmeans_until_converge(data, numClusters, tol=1e-6, max_iters = 10000):
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
    
    clusters = [[] for _ in range(numClusters)]
    for i, label in enumerate(labels):
        clusters[label].append(i + 1)
    return clusters

def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis = 1, keepdims = True)
    return matrix / np.where(norms == 0, 1, norms)

# Function to create 
def makeTestData() -> list[Node]:
    node_1 = Node(1)
    node_2 = Node(2)
    node_3 = Node(3)
    node_4 = Node(4)
    node_5 = Node(5)
    node_6 = Node(6)

    Connection(0.8, node_1, node_2)
    Connection(0.6, node_1, node_3)
    Connection(0.1, node_1, node_4)

    Connection(0.9, node_2, node_3)

    Connection(0.2, node_3, node_6)

    Connection(0.7, node_4, node_6)
    Connection(0.6, node_4, node_5)
            
    Connection(0.8, node_5, node_6)

    return [node_1, node_2, node_3, node_4, node_5, node_6]


if __name__ == "__main__":
    
    nodes = make_random_graph(9, 15)
    # nodes = makeTestData()
    a = convert_to_np_matrix(nodes)
    # print(a)

    # Construct normalized Laplacian
    degree = np.sum(a, axis=1)
    degree[degree == 0] = 1e-10  # Avoid division by zero

    D = np.diag(degree.astype(float))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    l = np.eye(len(a)) - D_inv_sqrt @ a @ D_inv_sqrt

    # Compute and sort eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(l)  # Use eigh for symmetric Laplacian
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]


    # Determine number of clusters from eigengap
    num_clusters = findClusterAmount(eigenvalues) + 1

    # Normalize eigenvectors row-wise
    norm_eigenvectors = normalize_rows(eigenvectors[:, :num_clusters])

    # Run k-means
    clusters = kmeans_until_converge(norm_eigenvectors, num_clusters)

    print(f"Clusters: {clusters}")
    print(f"Detected number of clusters: {num_clusters}")

    visualize_graph(a)
    
