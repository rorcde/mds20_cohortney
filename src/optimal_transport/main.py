import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from optimal_transport.trainer import Trainer
from optimal_transport.data_preprocessing import get_features
from src.DMHP.metrics import consistency, purity
import pandas as pd
import numpy as np
import torch

np.set_printoptions(threshold=10000)
torch.set_printoptions(threshold=10000)

def run_exp(path, ext, datetime, n_gamma_steps, min_partition, n_partitions, model, optimizer, criterion, device, in_channels, n_classes, lr, n_runs, with_gt = False, type_=None, **kwargs):
    X, steps = get_features(path, ext, datetime, n_gamma_steps = n_gamma_steps, min_partition = min_partition, n_partitions = n_partitions, type_=type_)
    X = [j.to(device) for j in X]
    
    if with_gt:
        gt = pd.read_csv(path+'/'+'clusters.csv')['cluster_id'].to_numpy()
        gt = torch.LongTensor(gt)
    else:
        gt = None

    selflabels = []
    purities = []
    for i in range(n_runs):
        print(f'============= RUN {i+1} ===============')
        Net = model(in_channels, len(steps), n_classes, steps, 64).to(device)
        optim = optimizer(Net.parameters(), lr = lr)
        trainer = Trainer(Net, optim, criterion, device,\
                          X, n_classes, gt = gt, **kwargs)
        s, p, sl = trainer.train()
        selflabels.append(torch.LongTensor(sl[0]))
        if with_gt:
            purities.append(p)
            print ('preds:', selflabels[i])
            print (f'Purity: {p:.4f}')
            
    if with_gt:
        print ('reals:', gt)
    
    print (selflabels)
    cons = consistency(selflabels)
    print(f'\nConsistency: {cons}')
        
    if with_gt:
        pur_val_mean = np.mean([purity(x, gt) for x in selflabels])
        pur_val_std = np.std([purity(x, gt) for x in selflabels])
        print(f'\nPurity: {pur_val_mean}+-{pur_val_std}')
    
    return selflabels, purities
    