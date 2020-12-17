from tick.hawkes import SimuHawkesExpKernels

def simulate_hawkes(n_nodes, n_realiz, end_time, dt, max_jumps=1000, seed=None, adj_density=0.25):
    """
    Simulate Hawkes process realizations
    
    Args:
        n_nodes (int)       - dimension of Hawkes process
        n_realiz (int)      - number of realizations
        end_time (float)    - right edge of interval where timestamps are distributed
        dt (float)          - delta to track intensity
        max_jumps (int)     - maximum number of jumps in Hawkes process
        seed (int)          - random seed
        adj_density (float) - density of the adjacency matrix of Hawkes process intensity
        
    Return:
        tss (List)          - list of realizations
    """
    
    tss = []

    baselines = np.random.rand(n_nodes) / n_nodes
    decays = 5 * np.random.rand(n_nodes, n_nodes)
    adjacency = spsp.rand(n_nodes, n_nodes, density=adj_density).toarray()

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