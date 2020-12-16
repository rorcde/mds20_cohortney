import torch

def sinkhorn(P, r, c, lmbd=0.05, eps = 1e-3):
    P = -P
    Q = torch.exp(- lmbd * P).T
    Q /= torch.sum(Q)
    K, N = Q.shape
    u = torch.zeros(K)
    res = []
    res.append(-torch.sum(torch.trace(Q.T @ P.T)))
    while True:
        Q_prev = Q.clone()
        u = torch.sum(Q, dim=1)
        Q *= (c / u).unsqueeze(1)
        Q *= (r / torch.sum(Q, dim=0)).unsqueeze(0)
        res.append(-torch.sum(torch.trace(Q.T @ P.T)))
        if torch.sum(((Q-Q_prev)/Q_prev)**2) <eps:
            break
    return Q.T, res