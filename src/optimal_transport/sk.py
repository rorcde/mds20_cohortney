import torch

def sinkhorn(P, r, c, lmbd=10, eps = 1e-3, max_iters = 100):
    P = -P
    Q = torch.exp(- lmbd * P).T
    Q /= torch.sum(Q)
    K, N = Q.shape
    u = torch.zeros(K)
    counter = 0
    while True:
        if counter == max_iters:
            break
        counter+=1
        Q_prev = Q.clone()
        u = torch.sum(Q, dim=1)
        Q *= (c / u).unsqueeze(1)
        Q *= (r / torch.sum(Q, dim=0)).unsqueeze(0)
        if torch.sum(((Q-Q_prev)/Q_prev)**2) <eps:
            break
    return Q.T

def get_labels(Q):
    return torch.argmax(Q, dim = 0)