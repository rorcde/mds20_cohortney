import torch

def sinkhorn(P, r, c, lmbd=0.05, eps = 1e-3, max_iters = 100):
    P = -P
    Q = torch.exp(- lmbd * P).T
    Q /= torch.sum(Q)
    K, N = Q.shape
    u = torch.zeros(K)
    res = []
    res.append(-torch.sum(torch.trace(Q.T @ P.T)))
    counter = 0
    while True:
        if counter == max_iters:
            break
        counter+=1
        Q_prev = Q.clone()
        u = torch.sum(Q, dim=1)
        Q *= (c / u).unsqueeze(1)
        Q *= (r / torch.sum(Q, dim=0)).unsqueeze(0)
        res.append(-torch.sum(torch.trace(Q.T @ P.T)))
        if torch.sum(((Q-Q_prev)/Q_prev)**2) <eps:
            break
    return Q.T, res

def get_labels(Q):
    return torch.argmax(Q, dim = 0)