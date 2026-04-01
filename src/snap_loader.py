"""
snap_loader.py
--------------
Phase 2C: Stanford SNAP P2P network loader and preprocessor.

Downloads and processes the Gnutella P2P network datasets:
  - p2p-Gnutella08 (6301 nodes, 20777 edges)
  - p2p-Gnutella31 (62586 nodes, 147892 edges)

Since these networks are too large for full SDE simulation, we:
  1. Download and parse the full graph
  2. Extract the largest weakly connected component
  3. Sample a representative connected subgraph of ~n_sample nodes
     using a breadth-first traversal from a high-degree seed
  4. Preserve topological properties (power-law degree distribution)

We also report full-graph topological statistics for Table in paper.
"""

import numpy as np
import os, sys, gzip, urllib.request, io
import networkx as nx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'snap')

SNAP_DATASETS = {
    'Gnutella08': {
        'url':  'https://snap.stanford.edu/data/p2p-Gnutella08.txt.gz',
        'file': 'p2p-Gnutella08.txt.gz',
        'n_full': 6301,
        'm_full': 20777,
    },
    'Gnutella31': {
        'url':  'https://snap.stanford.edu/data/p2p-Gnutella31.txt.gz',
        'file': 'p2p-Gnutella31.txt.gz',
        'n_full': 62586,
        'm_full': 147892,
    },
}


def download_snap(dataset_name, verbose=True):
    """Download a SNAP dataset to DATA_DIR if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    info     = SNAP_DATASETS[dataset_name]
    filepath = os.path.join(DATA_DIR, info['file'])

    if os.path.exists(filepath):
        if verbose:
            print(f"  Already downloaded: {filepath}")
        return filepath

    if verbose:
        print(f"  Downloading {dataset_name} from {info['url']} ...")
    try:
        urllib.request.urlretrieve(info['url'], filepath)
        if verbose:
            size_mb = os.path.getsize(filepath) / 1e6
            print(f"  Downloaded {size_mb:.1f} MB → {filepath}")
    except Exception as e:
        if verbose:
            print(f"  Download failed: {e}")
        return None

    return filepath


def parse_snap_edgelist(filepath, verbose=True):
    """Parse a SNAP .txt.gz edge list file into a NetworkX DiGraph."""
    if filepath is None or not os.path.exists(filepath):
        return None

    G = nx.DiGraph()
    opener = gzip.open if filepath.endswith('.gz') else open

    try:
        with opener(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    G.add_edge(int(parts[0]), int(parts[1]))
    except Exception as e:
        if verbose:
            print(f"  Parse error: {e}")
        return None

    if verbose:
        print(f"  Parsed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


def extract_subgraph(G, n_sample=500, seed=42, verbose=True):
    """
    Extract a connected subgraph of size ~n_sample.

    Strategy: BFS from the highest-degree node, collect n_sample nodes.
    This preserves hub structure of scale-free networks.
    """
    # Convert to undirected for topological analysis
    G_undir = G.to_undirected()

    # Largest connected component
    lcc = max(nx.connected_components(G_undir), key=len)
    G_lcc = G_undir.subgraph(lcc).copy()

    if verbose:
        print(f"  LCC size: {G_lcc.number_of_nodes()} nodes, {G_lcc.number_of_edges()} edges")

    # BFS from highest-degree node
    rng    = np.random.default_rng(seed)
    degrees = dict(G_lcc.degree())
    start   = max(degrees, key=degrees.get)   # hub node

    visited = []
    queue   = [start]
    seen    = {start}

    while queue and len(visited) < n_sample:
        node = queue.pop(0)
        visited.append(node)
        neighbors = list(G_lcc.neighbors(node))
        rng.shuffle(neighbors)
        for nb in neighbors:
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)

    # If BFS doesn't give enough, pad with random nodes from LCC
    if len(visited) < n_sample:
        remaining = list(set(G_lcc.nodes()) - set(visited))
        rng.shuffle(remaining)
        visited.extend(remaining[:n_sample - len(visited)])

    sub_nodes = visited[:n_sample]
    G_sub     = G_lcc.subgraph(sub_nodes).copy()

    # Ensure connectivity (take LCC of subgraph)
    if not nx.is_connected(G_sub):
        sub_lcc = max(nx.connected_components(G_sub), key=len)
        G_sub   = G_sub.subgraph(sub_lcc).copy()

    if verbose:
        print(f"  Subgraph: {G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges")

    # Relabel to 0..n-1
    G_sub = nx.convert_node_labels_to_integers(G_sub)

    return G_sub


def graph_topology_stats(G, name=''):
    """Compute topological statistics for a graph."""
    L     = nx.laplacian_matrix(G).toarray().astype(float)
    evals = np.sort(np.linalg.eigvalsh(L))
    lambda2 = float(evals[1]) if len(evals) > 1 else 0.0

    degrees  = [d for _, d in G.degree()]
    cc       = nx.average_clustering(G)

    stats = {
        'name':             name,
        'n':                G.number_of_nodes(),
        'm':                G.number_of_edges(),
        'avg_degree':       float(np.mean(degrees)),
        'max_degree':       int(max(degrees)),
        'std_degree':       float(np.std(degrees)),
        'lambda2':          lambda2,
        'clustering':       float(cc),
    }
    return stats, L


def extract_random_subgraph(G, n_sample=500, seed=42, min_lcc=400, max_attempts=20, verbose=True):
    """
    Extract a connected subgraph by random node sampling.

    Strategy: uniformly sample n_sample nodes, take the induced subgraph,
    extract its largest connected component. If LCC < min_lcc, resample.
    This gives a less biased structural sample than hub-seeded BFS.
    """
    G_undir = G.to_undirected()
    lcc     = max(nx.connected_components(G_undir), key=len)
    G_lcc   = G_undir.subgraph(lcc).copy()
    nodes   = list(G_lcc.nodes())

    rng = np.random.default_rng(seed)

    for attempt in range(max_attempts):
        sample = rng.choice(nodes, size=min(n_sample, len(nodes)), replace=False)
        G_ind  = G_lcc.subgraph(sample).copy()
        if not nx.is_connected(G_ind):
            sub_lcc  = max(nx.connected_components(G_ind), key=len)
            G_ind    = G_ind.subgraph(sub_lcc).copy()
        if G_ind.number_of_nodes() >= min_lcc:
            break
        rng = np.random.default_rng(seed + attempt + 1)
    else:
        if verbose:
            print(f"  Warning: random subgraph LCC={G_ind.number_of_nodes()} < {min_lcc}")

    G_ind = nx.convert_node_labels_to_integers(G_ind)
    if verbose:
        print(f"  Random subgraph: {G_ind.number_of_nodes()} nodes, "
              f"{G_ind.number_of_edges()} edges")
    return G_ind


def load_snap_networks(n_sample=500, verbose=True):
    """
    Download, parse, and subsample both SNAP P2P networks with two strategies.

    Returns dict:
      {
        'Gnutella08': {
            'bfs':    {'G': G_bfs,    'L': L_bfs,    'stats': stats_bfs},
            'random': {'G': G_random, 'L': L_random, 'stats': stats_random},
            'full_stats': {...},
        },
        'Gnutella31': { ... }
      }
    """
    results = {}

    for name, info in SNAP_DATASETS.items():
        if verbose:
            print(f"\n── {name} ──────────────────────────────")

        filepath = download_snap(name, verbose=verbose)
        G_full   = parse_snap_edgelist(filepath, verbose=verbose)

        if G_full is None:
            if verbose:
                print(f"  Could not load {name}, skipping.")
            continue

        full_stats = {
            'n': G_full.number_of_nodes(),
            'm': G_full.number_of_edges(),
        }

        # BFS hub-seeded subgraph (existing strategy)
        if verbose:
            print(f"  [BFS] hub-seeded subgraph:")
        G_bfs      = extract_subgraph(G_full, n_sample=n_sample, verbose=verbose)
        bfs_stats, L_bfs = graph_topology_stats(
            G_bfs, name=f'{name}_bfs_{G_bfs.number_of_nodes()}')
        if verbose:
            print(f"  BFS stats: n={bfs_stats['n']}, m={bfs_stats['m']}, "
                  f"λ₂={bfs_stats['lambda2']:.4f}, CC={bfs_stats['clustering']:.4f}")

        # Random node-sampled subgraph (new strategy)
        if verbose:
            print(f"  [Random] node-sampled subgraph:")
        G_rand      = extract_random_subgraph(G_full, n_sample=n_sample, verbose=verbose)
        rand_stats, L_rand = graph_topology_stats(
            G_rand, name=f'{name}_rand_{G_rand.number_of_nodes()}')
        if verbose:
            print(f"  Random stats: n={rand_stats['n']}, m={rand_stats['m']}, "
                  f"λ₂={rand_stats['lambda2']:.4f}, CC={rand_stats['clustering']:.4f}")

        results[name] = {
            'bfs': {
                'G':     G_bfs,
                'L':     L_bfs,
                'stats': bfs_stats,
            },
            'random': {
                'G':     G_rand,
                'L':     L_rand,
                'stats': rand_stats,
            },
            'full_stats': full_stats,
        }

    return results


if __name__ == '__main__':
    nets = load_snap_networks(n_sample=500, verbose=True)
    for name, data in nets.items():
        print(f"\n{name}:")
        print(f"  Full: n={data['full_stats']['n']}, m={data['full_stats']['m']}")
        print(f"  Sub : n={data['stats']['n']}, m={data['stats']['m']}")
        print(f"  λ₂  : {data['stats']['lambda2']:.4f}")
