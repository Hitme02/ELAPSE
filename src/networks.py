"""
networks.py
-----------
Generates the three network topologies used throughout the QUORUM simulations.
Each returns a NetworkX graph + its weighted Laplacian matrix.
"""

import numpy as np
import networkx as nx


def make_erdos_renyi(n, p=0.15, seed=42):
    """
    Erdos-Renyi random graph G(n, p).
    High spectral gap (lambda_2 large) -> fast diffusion -> fast entropy rise.
    """
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    # Ensure connected
    while not nx.is_connected(G):
        seed += 1
        G = nx.erdos_renyi_graph(n, p, seed=seed)
    return G, _laplacian(G)


def make_barabasi_albert(n, m=3, seed=42):
    """
    Barabasi-Albert scale-free graph.
    Hub structure -> heterogeneous spreading -> intermediate lambda_2.
    """
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    return G, _laplacian(G)


def make_watts_strogatz(n, k=6, p=0.1, seed=42):
    """
    Watts-Strogatz small-world graph.
    Low lambda_2 -> slow diffusion -> slow entropy rise -> late mortality.
    """
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    while not nx.is_connected(G):
        seed += 1
        G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    return G, _laplacian(G)


def _laplacian(G):
    """Unnormalised graph Laplacian as numpy array."""
    return nx.laplacian_matrix(G).toarray().astype(float)


def fiedler_value(L):
    """Second smallest eigenvalue of L (spectral gap lambda_2)."""
    eigenvalues = np.sort(np.linalg.eigvalsh(L))
    return eigenvalues[1]


def get_all_networks(n):
    """Returns dict of {name: (G, L)} for all three topologies."""
    return {
        "Erdos-Renyi":       make_erdos_renyi(n),
        "Barabasi-Albert":   make_barabasi_albert(n),
        "Watts-Strogatz":    make_watts_strogatz(n),
    }
