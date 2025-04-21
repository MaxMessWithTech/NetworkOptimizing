from scipy.linalg import fractional_matrix_power
import numpy as np
from node import Node
from connection import Connection
import algo
import array
from sklearn.cluster import KMeans

import create # import visualize_graph, make_random_graph, convert_to_np_matrix

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
    
    nodes = create.make_random_graph(9, 15)
    # nodes = makeTestData()
    a = create.convert_to_np_matrix(nodes)
    # print(a)

    clusters, num_clusters = algo.compute_connectivity(matrix=a)

    print(f"Clusters: {clusters}")
    print(f"Detected number of clusters: {num_clusters}")

    create.visualize_graph(a)
    
