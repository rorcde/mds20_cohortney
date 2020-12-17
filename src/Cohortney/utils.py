import torch
import numpy as np


def make_grid(gamma, T_b, T_h, N, n):
    grid = []
    for i in range(N):
        a = gamma**i * T_b
        if (a <= T_h):
            grid.append(a)
            
        else:
            break
    grid = np.array(grid)

    return grid
    