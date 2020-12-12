import torch
import numpy as np
import scipy as sp
import scipy.sparse as spsp
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser

from tick.hawkes import SimuHawkesExpKernels, SimuHawkesSumExpKernels
from tick.plot import plot_point_process


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--n_clusters', type=int, default=3)
    parser.add_argument('--n_nodes', type=int, default=5)
    parser.add_argument('--n_realiz_per_cluster', type=int, default=100)
    parser.add_argument('--n_decays', type=int, default=3)
    parser.add_argument('--end_time', type=float, default=100)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--max_jumps', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--adj_density', type=float, default=0.25)
    parser.add_argument('--seed', type=int)
    
    args = parser.parse_args()

    return args


def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def simulate_hawkes(n_nodes, n_decays, n_realiz, end_time, dt, max_jumps=1000, seed=None, adj_density=0.25):
    tss = []

    baselines = np.random.rand(n_nodes) / n_nodes
    #baselines = spsp.rand(1, n_nodes, density=0.5).toarray()[0] / n_nodes
    decays = 5 * np.random.rand(n_nodes, n_nodes)
    adjacency = spsp.rand(n_nodes, n_nodes, density=adj_density).toarray()
    # Simulation

    for i in range(n_realiz):
        seed_ = seed + i if seed is not None else None
        hawkes = SimuHawkesExpKernels(
            baseline=baselines, decays=decays, adjacency=adjacency, seed=seed_)
        hawkes.adjust_spectral_radius(0.8)
        hawkes.max_jumps = max_jumps

        hawkes.end_time = end_time
        hawkes.verbose = False

        hawkes.track_intensity(dt)
        hawkes.simulate()
        tss.append(hawkes.timestamps)
    
    return tss 


def simulate_clusters(n_clusters, n_nodes, n_decays, n_realiz, end_time, dt, max_jumps, seed=None, adj_density=None):
    clusters = []
    for i in range(n_clusters):
        seed_ = seed + i if seed is not None else None
        clusters.append(simulate_hawkes(n_nodes, n_decays, n_realiz, end_time, dt, max_jumps, seed_, adj_density))
    
    return clusters


def convert_seq_to_df(timestamps):
    ts = []
    cs = []

    for c, tc in enumerate(timestamps):
        cs += [c]*len(tc)
        ts += list(tc)
    s = list(zip(ts, cs))
    s = list(sorted(s, key=lambda x: x[0]))
    s = np.array(s)
    df = pd.DataFrame(data=s, columns=['time', 'event'])
    
    return df


def convert_clusters_to_dfs(clusters):
    dfs = []
    cluster_ids = []

    for cl_id, cluster in enumerate(clusters):
        cluster_ids += [cl_id] * len(cluster)
        for realiz in cluster:
            df = convert_seq_to_df(realiz)
            dfs.append(df)

    return dfs, cluster_ids


if __name__ == '__main__':
    args = parse_arguments()
    if args.seed is not None:
        random_seed(args.seed)
    print('Simulating...')
    clusters = simulate_clusters(args.n_clusters, args.n_nodes, 
                args.n_decays, args.n_realiz_per_cluster, args.end_time, args.dt, args.max_jumps, args.seed, args.adj_density)
    dfs, cluster_ids = convert_clusters_to_dfs(clusters)
    print('Saving...')
    save_dir = Path(Path(__file__).parent.absolute(), 'simulated_Hawkes', args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    for i, df in enumerate(dfs):
        df.to_csv(Path(save_dir, f'{i+1}.csv').open('w'))

    pd.DataFrame(data=np.array(cluster_ids), columns=['cluster_id']).to_csv(Path(save_dir, f'clusters.csv').open('w'))
    print('Finished.')
