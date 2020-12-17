import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from dataPreprocessing.cohortney import load_data, arr_func, multiclass_fws_array, events_tensor

import torch

def get_features(path, ext, datetime, gamma= 1.04, n_gamma_steps = 5, min_partition = 0, n_partitions = 5, verbose = True):
    ss, Ts, class2idx, user_list = load_data(path, ext = ext, datetime = datetime)
    T_h = max(Ts)
    X = []
    steps = []
    for i in range(n_gamma_steps-1, -1, -1):
        T_j = T_h/gamma**i
        for j in range(min_partition, n_partitions):
            if verbose:
                print(n_gamma_steps-1 - i, j)
            delta_T = torch.range(0, T_j, T_j/2**j)
            steps.append(len(delta_T)-1)
            _,  events_fws_mc = arr_func(user_list, 300, delta_T, multiclass_fws_array)
            mc_batch = events_tensor(events_fws_mc)
            X.append(mc_batch)
    return X, steps