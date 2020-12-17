import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from optimal_transport.trainer import Trainer
from optimal_transport.data_preprocessing import get_features
import pandas as pd
import torch

def run_exp(path, ext, datetime, n_gamma_steps, n_partitions, model, optimizer, criterion, device, in_channels, n_classes, lr, with_gt = False, **kwargs):
    X, steps = get_features(path, ext, datetime, n_gamma_steps = n_gamma_steps, n_partitions = n_partitions)
    X = [j.to(device) for j in X]
    Net = model(in_channels, n_gamma_steps*n_partitions, n_classes, steps, 64).to(device)
    optim = optimizer(Net.parameters(), lr = lr)
    if with_gt:
        gt = pd.read_csv(path+'/'+'clusters.csv')['cluster_id'].to_numpy()
        gt = torch.LongTensor(gt)
    else:
        gt = None
    trainer = Trainer(Net, optim, criterion, device,\
                 X, n_classes, gt = gt)
    trainer.train()
    