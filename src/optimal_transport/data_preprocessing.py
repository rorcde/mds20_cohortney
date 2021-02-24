import os
import sys
import inspect
import torch
import numpy as np

from Cohortney.cohortney import arr_func, multiclass_fws_array, events_tensor
from Cohortney.data_utils import load_data

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)


def get_features(path, ext, datetime, gamma=1.04, n_gamma_steps=5, min_partition=0, n_partitions=5, verbose=False, type_=None):
    ss, Ts, class2idx, user_list = load_data(path, ext = ext, datetime = datetime, type_=type_)
    T_h = max(Ts)
    X = []
    steps = []
    for i in range(n_gamma_steps-1, -1, -1):
        T_j = T_h/gamma**i
        for j in range(min_partition, n_partitions):
            if verbose:
                print(n_gamma_steps-1 - i, j)
            delta_T = np.arange(0, T_j+0.0001, T_j/2**j)
            while delta_T[-1]>T_j:
                delta_T = delta_T[:-1]
            steps.append(len(delta_T)-1)
            _,  events_fws_mc = arr_func(user_list, 300, delta_T, multiclass_fws_array)
            mc_batch = events_tensor(events_fws_mc)
            X.append(mc_batch)
    return X, steps