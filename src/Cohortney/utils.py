import torch
import numpy as np


def fws(p, t1, t2):
    n = sum(list(map(int, (p >= t1) & (p <= t2))))
    return min(int(np.log2(n+1)), 9)


#fws array as a string (for cohortney)
def fws_array(p, array):
    fws_array = ''
    for i in range(1, len(array)):
        fws_array += str(fws(p, array[i-1], array[i]))
    # fws_array = tuple(fws_array)
    return fws_array


#fws array as array for AE
def fws_numerical_array(p, array):
    fws_array = []
    for i in range(1, len(array)):
        fws_array.append(fws(p, array[i-1], array[i]))
    # fws_array = tuple(fws_array)
    return fws_array


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


def arr_func(events, T_j, Delta_T):
    events_fws = dict()
    for p_k in events:
        fws_val =  fws_numerical_array(p_k, Delta_T)

        p_k1 = tuple(p_k)
        if p_k1 not in events_fws.keys():
            events_fws[p_k1] = []
            events_fws[p_k1].append(fws_val)
        else:
            events_fws[p_k1].append(fws_val)

    array = []
    for val in events_fws.values():
        array.append(list(val[0]))
    return array, events_fws
    