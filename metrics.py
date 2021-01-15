import math
from collections import defaultdict
import numpy as np
from util import *
from sklearn.metrics import pairwise_distances_argmin
from scipy.stats import wasserstein_distance


def compute_centrality(graph):
    return nx.eigenvector_centrality_numpy(graph, weight='weight')


def find_node_roles(graph: nx.Graph, attribute='weight'):
    # Build dictionary mapping node_ids to adjacent edge weights
    edge_weights = defaultdict(list)
    # Loop through all nodes, add attribute to each that is the sum of all adjacent edge weights
    for node in graph.nodes():
        for neighbor in graph.neighbors(node):
            weight = graph.get_edge_data(node, neighbor)[attribute]
            if weight > 0:
                if attribute == 'weight':
                    edge_weights[node].append(math.log(weight))
                else:
                    edge_weights[node].append(weight)
    # Convert each node array of edges to histogram; Find global min and max values of weights
    min_weight = min([min(weights) for node, weights in edge_weights.items()])
    max_weight = max([max(weights) for node, weights in edge_weights.items()])
    # Build histograms
    edge_weights_his = {}
    for node, weights in edge_weights.items():
        hist = np.histogram(weights, bins=13, range=(min_weight, max_weight))
        edge_weights_his[node] = list(hist[0])
    # Finish
    nodes, histograms = zip(*edge_weights_his.items())
    centers, labels = find_clusters(np.array(histograms), 5)
    node_roles = dict(zip(nodes, labels))
    # Return
    return node_roles


def find_clusters(X, n_clusters, seed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(seed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        # print(centers)
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers, metric=wasserstein_distance)
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        # 2c. Check for convergence
        if np.all(centers == new_centers): break
        centers = new_centers
        # Break if values are invalid
        if np.any(np.isnan(centers)) or np.any(np.isinf(centers)): break
    return centers, labels


