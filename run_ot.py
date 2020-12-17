import torch
import torch.nn as nn
from src.optimal_transport.data_preprocessing import get_features
from models.OTModel.ot import CNN_Filters
from src.optimal_transport.main import run_exp
from src.DMHP.metrics import consistency
import numpy as np

path = input('path')
ext = input('ext')
datetime = input('datetime')
n_gamma = input('n_gamma')
min_partition = input('min_partition')
n_partition = input('n_partition')
in_channels = input('in_channels')
n_classes = input('n_classes')
lr = input('lr')
n_runs = input('n_runs')
with_gt = input('with_gt')
max_epochs = input('max_epochs')
devc = '0'
device = torch.device('cuda:' + devc) if torch.cuda.is_available() else torch.device('cpu')
print(device)

s, p = run_exp(path, ext, datetime, n_gamma, min_partition, n_partition, CNN_Filters, torch.optim.Adam, nn.CrossEntropyLoss(), device, in_channels, n_classes, lr, n_runs, with_gt = with_gt, max_epochs = max_epoch)

if p[0]!=None:
    print('Purity: ', np.mean(p), 'pm',np.std(p))
print('Consistency: ', consistency(s))