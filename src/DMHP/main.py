from pathlib import Path
from zipfile import ZipFile
import pandas as pd
import argparse
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

from src.DMHP.HP import PointProcessStorage, DirichletMixtureModel, EM_clustering, tune_basis_fn
from src.DMHP.metrics import consistency, purity
from src.Cohortney.data_utils import load_data


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
    ss, Ts, class2idx, _ = load_data(Path(args.data_dir), maxsize=args.maxsize, maxlen=args.maxlen, ext=args.ext, datetime=not args.not_datetime, type_=args.type)
    
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
    nlls = torch.zeros(args.nruns, niter)

    assigned_labels = []
    results = {}
    for i in range(args.nruns):
        print(f'============= RUN {i+1} ===============')

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
        
    cons = consistency(assigned_labels)

    print(f'Consistency: {cons:.4f}\n')
    results['consistency'] = cons
   
    if gt_ids is not None:
        pur_val_mean = np.mean([purity(x, gt_ids) for x in labels])
        pur_val_std = np.std([purity(x, gt_ids) for x in labels])
        print(f'Purity: {pur_val_mean:.4f}+-{pur_val_std:.4f}')
    

if __name__ == "__main__":
    args = parse_arguments()
    if args.seed is not None:
        random_seed(args.seed)
    main(args)