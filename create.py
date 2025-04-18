from node import Node
from connection import Connection
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import datetime as dt
import pytz

def visualize_graph(matrix: np.array) -> None:
    adj_matrix = np.array(matrix)
    graph = nx.from_numpy_array(adj_matrix)

    # Relabel from 0-based to 1-based
    mapping = {i: i + 1 for i in range(len(graph))}
    graph = nx.relabel_nodes(graph, mapping)

    pos = nx.spring_layout(graph, seed=42)

    # Debugging info
    print("Adjacency Matrix:\n", adj_matrix)
    print("Graph edges with weights:")
    for u, v, d in graph.edges(data=True):
        print(f"{u} -- {v} (weight={d['weight']:.3g})")

    plt.clf()  # Clear any existing plots

    nx.draw(
        graph, pos, with_labels=True,
        node_color='skyblue',
        node_size=800,
        edge_color='gray',
        linewidths=1,
        font_weight='bold'
    )

    # Format weights
    edge_weights = nx.get_edge_attributes(graph, 'weight')
    formatted_weights = {k: f"{v:.2g}" for k, v in edge_weights.items()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=formatted_weights, font_size=10)

    # Save and show
    cur_time = dt.datetime.now(tz=pytz.timezone('US/Central')).strftime("%Y-%m-%d %H-%M-%S")
    filename = f"Graph {cur_time}.png"

    plt.title("Graph Visualization with 1-based Node Labels", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

"""
def convert_to_np_matrix(matrix:list[list[Connection]]) -> np.matrix:
    fixed_matrix = list()
    for i in range(len(matrix)):
        newRow = list()

        for j in range(len(matrix[i])):
            val = matrix[i][j]

            if isinstance(val, Connection):
                newRow.append(val.weight)
            elif val is None:
                newRow.append(0)
            else:
                print(val)
        fixed_matrix.append(newRow)
    return fixed_matrix
"""

def convert_to_np_matrix(nodes: list[Node]) -> np.matrix:
    matrix = [0] * len(nodes)

    for node in nodes:
        row = [0] * len(nodes)
        for conn in node.getConnections():
            other_node = conn.getOtherNode(node)
            row[other_node.index - 1] = conn.weight

        matrix[node.index - 1] = row
    
    return np.array(matrix)


def make_random_graph_OL(node_count:int, connection_count:int) -> list[Node]:
    print("\r\nCREATING A WEIGHTED GRAPH\r\n")
    nodes = list()
    matrix = list(list())

    # Make All Nodes
    for i in range(node_count):
        nodes.append( Node( index = (i + 1) ) )
        matrix.append([None] * node_count)

    for i in range(connection_count):
        a = 0
        b = 0
        while True:
            a = random.randint(0, node_count - 1)
            b = random.randint(0, node_count - 1)


            # print(f"\rAttempting to place #{i + 1} between {a} & {b}... ", end="")
            if a != b and matrix[a][b] is None:
                # print(f"Placed.", end="\r\n")
                break
                
        conn = Connection(random.random(), nodes[a], nodes[b])
        matrix[a][b] = conn
        matrix[b][a] = conn

    print(matrix)
    print(f"\r\nSUCCESSFULLY created graph with {node_count} nodes and {connection_count} connections\r\n")

    return nodes
        
def make_random_graph(node_count: int, connection_count: int) -> list[Node]:
    print("\nCREATING A CONNECTED WEIGHTED GRAPH\n")

    if connection_count < node_count - 1:
        raise ValueError("To ensure connectivity, connection_count must be at least node_count - 1")

    # Create nodes with 1-based indices
    nodes = [Node(index=i + 1) for i in range(node_count)]
    matrix = [[None] * node_count for _ in range(node_count)]
    edges = set()

    # Step 1: Build a spanning tree to ensure connectivity
    available = list(range(node_count))  # 0-based internally
    connected = [available.pop(random.randint(0, len(available) - 1))]

    while available:
        from_idx = random.choice(connected)
        to_idx = available.pop(random.randint(0, len(available) - 1))
        weight = random.random()

        # Adjust for 1-based indexing in nodes
        conn = Connection(weight, nodes[from_idx], nodes[to_idx])
        matrix[from_idx][to_idx] = conn
        matrix[to_idx][from_idx] = conn
        edges.add(tuple(sorted((from_idx + 1, to_idx + 1))))  # 1-based tracking
        connected.append(to_idx)

    # Step 2: Add remaining edges randomly
    while len(edges) < connection_count:
        a, b = random.sample(range(node_count), 2)
        edge_key = tuple(sorted((a + 1, b + 1)))  # 1-based keys

        if edge_key not in edges:
            weight = random.random()
            conn = Connection(weight, nodes[a], nodes[b])
            matrix[a][b] = conn
            matrix[b][a] = conn
            edges.add(edge_key)

    print(f"\nGraph with {node_count} nodes and {len(edges)} connections successfully created.\n")
    return nodes

if __name__ == "__main__":
    node_count = int(input("Nodes: "))
    connection_count = int(input("Connections: "))

    nodes = make_random_graph(node_count, connection_count)
    visualize_graph(convert_to_np_matrix(nodes))
    
    