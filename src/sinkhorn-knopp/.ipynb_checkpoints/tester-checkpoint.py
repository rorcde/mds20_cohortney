import torch
import matplotlib.pyplot as plt

from sk import *

N, K = 100, 5
P = torch.rand((K, N))
r = torch.ones(K)/K
c = torch.ones(N)/N
lmbd = 0.1
eps = 1e-3

#Q = sk_optimal_transport(P, r, c, N, K, lmbd, eps)
Q, res = sinkhorn(P, r, c, lmbd, eps)
plt.plot(res)
plt.savefig('test.jpg')