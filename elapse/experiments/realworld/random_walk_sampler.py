"""
random_walk_sampler.py
---------------------
Random-walk based network subgraph sampler.
Uses teleporting random walk (p_teleport=0.15) to get representative samples.
"""

import numpy as np
import networkx as nx


def random_walk_sample(G, n_target=500, p_teleport=0.15, min_lcc=400,
                        max_attempts=20, seed=42):
    """
    Sample n_target nodes from G via teleporting random walk.

    At each step:
      - With prob (1-p_teleport): move to random neighbour
      - With prob p_teleport: jump to random node

    Collect n_target unique visited nodes.
    Take induced subgraph, extract LCC.
    Retry up to max_attempts if LCC < min_lcc.

    Returns: G_sub (connected subgraph with >= min_lcc nodes), or
             largest attempt if all fail.
    """
    G_undir = G.to_undirected() if G.is_directed() else G
    lcc_nodes = max(nx.connected_components(G_undir), key=len)
    G_lcc = G_undir.subgraph(lcc_nodes).copy()
    nodes = list(G_lcc.nodes())

    if len(nodes) < min_lcc:
        return G_lcc

    rng = np.random.default_rng(seed)
    best_sub = None
    best_size = 0

    for attempt in range(max_attempts):
        # Start from random node
        current = rng.choice(nodes)
        visited = set([current])
        visited_order = [current]

        while len(visited) < n_target:
            if rng.random() < p_teleport or len(list(G_lcc.neighbors(current))) == 0:
                # Teleport to random node
                current = rng.choice(nodes)
            else:
                # Walk to random neighbour
                neighbours = list(G_lcc.neighbors(current))
                current = rng.choice(neighbours)

            if current not in visited:
                visited.add(current)
                visited_order.append(current)

        G_sub = G_lcc.subgraph(list(visited)[:n_target]).copy()

        # Extract LCC of subgraph
        if not nx.is_connected(G_sub):
            sub_lcc_nodes = max(nx.connected_components(G_sub), key=len)
            G_sub = G_sub.subgraph(sub_lcc_nodes).copy()

        if G_sub.number_of_nodes() >= min_lcc:
            G_sub = nx.convert_node_labels_to_integers(G_sub)
            return G_sub

        if G_sub.number_of_nodes() > best_size:
            best_size = G_sub.number_of_nodes()
            best_sub = G_sub

    print(f"Warning: best LCC size = {best_size} < {min_lcc}")
    best_sub = nx.convert_node_labels_to_integers(best_sub)
    return best_sub
