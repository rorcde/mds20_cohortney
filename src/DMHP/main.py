from pathlib import Path
from zipfile import ZipFile
import pandas as pd
import argparse
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import time
from sklearn.metrics.cluster import normalized_mutual_info_score
def random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
from HP import PointProcessStorage, DirichletMixtureModel, EM_clustering, tune_basis_fn
from metrics import consistency, purity
from data_utils import load_data
import tarfile
import pickle
import sys
import json

sys.path.append("..")

def random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='dir holding sequences as separate files')
    parser.add_argument('--maxsize', type=int, default=None, help='max number of sequences')
    parser.add_argument('--nmb_cluster', type=int, default=10, help='number of clusters')
    parser.add_argument('--maxlen', type=int, default=-1, help='maximum length of sequence')
    parser.add_argument('--ext', type=str, default='txt', help='extention of files with sequences')
    parser.add_argument('--not_datetime', action='store_true', help='if time values in event sequences are represented in datetime format')
    # hyperparameters for Cohortney
    parser.add_argument('--gamma', type=float, default=1.4)
    parser.add_argument('--Tb', type=float, default=7e-6)
    parser.add_argument('--Th', type=float, default=80)
    parser.add_argument('--N', type=int, default=2500)
    parser.add_argument('--n', type=int, default=4, help='n for partition')
    # hyperparameters for training
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--save_to', type=str, default="DMHP_Metrics", help='directory for saving metrics')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--nruns', type=int, default=1, help='number of trials')
    parser.add_argument('--type', type=str, default=None, help='if it is a')

    parser.add_argument('--result_path', type=str, help='path to save results')
    args = parser.parse_args()
    return args

np.set_printoptions(threshold=10000)
torch.set_printoptions(threshold=10000)

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ss, Ts, class2idx = load_data(Path(args.data_dir), maxsize=args.maxsize, maxlen=args.maxlen, ext=args.ext, datetime=not args.not_datetime, type_=args.type)
    
    gt_ids = None
    if Path(args.data_dir, 'clusters.csv').exists():
        gt_ids = pd.read_csv(Path(args.data_dir, 'clusters.csv'))['cluster_id'].to_numpy()
        gt_ids = torch.LongTensor(gt_ids)
    
    N = len(ss)
#     D = 5 # the dimension of Hawkes processes
    w = 1
    
#     basis_fs = [lambda x: torch.exp(- x**2 / (3.*(k+1)**2) ) for k in range(D)]
    
    not_tune = False
    if args.ext == 'txt':
        eps = 1e6
        not_tune = True
    else:
        eps = 1e5
    if args.type == 'booking1' or args.type == 'booking2':
        eps = 1e2
    basis_fs = tune_basis_fn(ss, eps=eps, not_tune=not_tune)
    D = len(basis_fs) # the dimension of Hawkes processes
    hp = PointProcessStorage(ss, Ts, basis_fs)

    C = len(class2idx)
    K = args.nmb_cluster

    niter = 10
    labels = torch.zeros(args.nruns, len(ss))
    info_score = np.zeros((K+1, K+1))
    nlls = torch.zeros(args.nruns, niter)
    times = np.zeros(args.nruns)

    assigned_labels = []
    results = {}
    for i in range(args.nruns):
        print(f'============= RUN {i+1} ===============')
        time_start = time.clock()
        Sigma = torch.rand(C, C, D, K)
        B = torch.rand(C, K)
        alpha = 1.

        model = DirichletMixtureModel(K, C, D, alpha, B, Sigma)
        print ('model ready')
        EM = EM_clustering(hp, model)
        print ('EM ready')
        r, nll_history, r_history = EM.learn_hp(niter=niter, ninner=[2,3,4,5,6,7] + (niter - 6)*[8])
        print ('learn_hp ready')

        labels[i] = r.argmax(-1)
        nlls[i] = torch.FloatTensor(nll_history)

        print ("preds:", labels[i])
        
        assigned_labels.append(labels[i])
#         if args.verbose:
#             print(f'Sizes of clusters: {", ".join([str((torch.tensor(labels[i]) == i).sum().item()) for i in range(args.nmb_cluster)])}\n')
        
        if gt_ids is not None:
            print(f'Purity: {purity(labels[i], gt_ids):.4f}')
          
            for k in range(1,K+1):
                info_score[k, 0] = k-1
                for j in range(1, K+1):
                    info_score[0, j] = j-1
                    ind = np.concatenate([np.argwhere(gt_ids == j-1), np.argwhere(gt_ids == k-1)], axis=1)[0]
                    a = labels[i].tolist()
                    b = gt_ids.tolist()
                    info_score[k, j] += normalized_mutual_info_score([b[i] for i in ind], [a[i] for i in ind])/args.nruns
        times[i] = time.clock() - time_start
    cons = consistency(assigned_labels)

    print(f'Consistency: {cons:.4f}\n')
    results['consistency'] = cons
    if gt_ids is not None:
        pur_val_mean = np.mean([purity(x, gt_ids) for x in labels])
        pur_val_std = np.std([purity(x, gt_ids) for x in labels])
        print(f'Purity: {pur_val_mean:.4f}+-{pur_val_std:.4f}')
        print(f'Normalized mutual info score: {info_score}')
    time_mean = np.mean(times)
    time_std = np.std(times)
    nll_mean = torch.mean(nlls)
    nll_std = torch.std(nlls)
    print(f'Mean run time: {time_mean:.4f}+-{time_std:.4f}')
    if (args.save_to is not None) and (gt_ids is not None):
        metrics = {
            "Purity": f'{pur_val_mean:.4f}+-{pur_val_std:.4f}' ,
            "Mean run time": f'{time_mean:.4f}+-{time_std:.4f}',
            "Normalized mutual info score:": f'{info_score}',
            "Predictive log likelihood:":f'{nll_mean.item():.4f}+-{nll_std.item():.4f}'
        }
        with open(Path(args.data_dir, args.save_to), "w") as f:
            json.dump(metrics, f, indent=4)
    else:
        metrics = {
            "Mean run time": f'{time_mean:.4f}+-{time_std:.4f}',
            "Predictive log likelihood:":f'{nll_mean.item():.4f}+-{nll_std.item():.4f}',
            "Predicted labels":f'{labels}'
        }
        with open(Path(args.data_dir, args.save_to), "w") as f:
            json.dump(metrics, f, indent=4)
            

if __name__ == "__main__":
    args = parse_arguments()
    if args.seed is not None:
        random_seed(args.seed)
    main(args)