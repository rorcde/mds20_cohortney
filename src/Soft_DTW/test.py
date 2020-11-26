import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

from soft_dtw import *

a = torch.randn(5, 4)
b = torch.randn(6, 4)

criterion = Soft_DTW(gamma=1.0, norm=True)
loss = criterion(a, b)
print (loss)