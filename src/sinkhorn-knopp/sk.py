import torch

def sk_optimal_transport(P, r, c, N, K, lmbd, eps):
    #unstable
    tmp = torch.exp(-lmbd * P)
    alpha = r
    beta = c
    alpha_prev = torch.zeros_like(r)
    err = 1
    while err > eps:
        beta_prev = beta.clone()
        alpha_prev = alpha.clone()
        alpha = 1./(tmp @ beta_prev)
        beta = 1./(alpha_prev @ tmp)
        err = torch.sum(((alpha - alpha_prev)/alpha)**2)
    return torch.diag(alpha) @ tmp @ torch.diag(beta)

def sinkhorn(P, r, c, lmbd=0.05, eps = 1e-3):
    #has bug, to be tested
    Q = torch.exp(- lmbd * P).T
    Q /= torch.sum(Q)
    K, N = Q.shape
    u = torch.zeros(K)
    res = []
    res.append(torch.sum(torch.trace(Q.T @ P.T)))
    while True:
        Q_prev = Q.clone()
        u = torch.sum(Q, dim=1)
        Q *= (c / u).unsqueeze(1)
        Q *= (r / torch.sum(Q, dim=0)).unsqueeze(0)
        res.append(torch.sum(torch.trace(Q.T @ P.T)))
        if torch.sum(((Q-Q_prev)/Q_prev)**2) <eps:
            break
    return Q.T, res