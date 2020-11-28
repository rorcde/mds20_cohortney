# This file contains PyTorch implementation of methods from A Dirichlet Mixture Model 
# of Hawkes Processes for Event Sequence Clustering
# https://arxiv.org/pdf/1701.09177.pdf

import torch
import numpy as np
from tqdm import tqdm, trange
from typing import List
from torch.nn import functional as F


def integral(f, left, right, npts=10000):
    grid = torch.FloatTensor(np.linspace(left.tolist(), right.tolist(), npts))
    out = f(grid)
    int_f = (out[0, :] / 2. + out[-1, :] / 2. + out[1:-1, :]).sum(0) * (grid[1, :] - grid[0, :])
    return int_f    


class PointProcessStorage:
    """
    Class for storing point process' realizations

    Args:
    - seqs - list of realizations, where each item is a tensor of size (L, 2)
    - Tns - tensor of Tn for each realization
    - basis_fs - list of basis functions, taking tensor as input

    """
    def __init__(self, seqs : List, Tns : torch.Tensor, basis_fs : List):
        self.N = len(seqs)
        self.seqs = seqs
        self.Tns = Tns
        self.basis_fs = basis_fs

    def __iter__(self):
        for n in range(len(self.seqs)):
            cs, ts = self.seqs[n][:, 1], self.seqs[n][:, 0]
            Tn = self.Tns[n]
            assert Tn > ts[-1]
            yield cs.long(), ts.float(), Tn


class DirichletMixtureModel:
    """
    Dirichlet Mixture Model for Hawkes Processes

    Args:
    - K - number of clusters
    - C - number of event classes
    - D - number of basis functions
    - alpha - parameter of Dirichlet distribution
    - B - parameter of Reyleight distribution - prior of \mu
    - Sigma - parameter of Exp distribution - prior of A

    """
    def __init__(self, K, C, D, alpha, B=None, Sigma=None):
        self.alpha = alpha
        self.K = K
        concentration = torch.FloatTensor([alpha /  float(K) for _ in range(K)])
        self.dirich = torch.distributions.dirichlet.Dirichlet(concentration)
        self.pi = self.dirich.sample()
        self.cat = torch.distributions.categorical.Categorical(self.pi)
        self.k_pi = self.cat.sample()
        self.B = B  # C x K
        self.rayleigh = torch.distributions.weibull.Weibull(2**0.5 * B, 2 * torch.ones_like(B))
        self.mu = self.rayleigh.sample()
        self.Sigma = Sigma
        self.exp = torch.distributions.exponential.Exponential((Sigma)**(-1))
        self.A = self.exp.sample() # C x C x D x K

    def p_mu(self, mu=None):
        mu = mu if mu is not None else self.mu
        return torch.exp(self.rayleigh.log_prob(mu))

    def p_A(self, A=None):
        A = A if A is not None else self.A
        return torch.exp(self.exp.log_prob(A))

    def p_pi(self, pi=None):
        pi = pi if pi is not None else self.pi
        return torch.exp(self.dirich.log_prob(pi))

    def e_logpi(self):
        """
        Returns expectation of logarithm of \pi
        """
        return (torch.digamma(torch.FloatTensor([self.alpha / self.K])) - torch.digamma(torch.FloatTensor([self.alpha]))).repeat(self.K)

    def e_mu(self):
        """
        Returns expectation of \mu
        """
        return (np.pi / 2)**.5 * self.B

    def e_A(self):
        """
        Returns expectation of \A
        """
        return self.Sigma

    def var_mu(self):
        """
        Returns variance of \mu
        """
        return (4 - np.pi) / 2 * self.B**2

    def var_A(self):
        """
        Returns variance of \A
        """
        return self.Sigma**2

    def update_A(self, A, Sigma):
        self.A = A
        self.Sigma = Sigma
        self.exp = torch.distributions.exponential.Exponential((Sigma)**(-1))

    def update_mu(self, mu, B):
        self.mu = mu
        self.B = B
        self.rayleigh = torch.distributions.weibull.Weibull(2**0.5 * B, 2 * torch.ones_like(B))

    def update_pi(self, pi):
        self.pi = pi


class EM_clustering:
    """
    Class for learning parameters of Hawkes Processes' clusters

    Args:
    - hp - PointProcessStorage object
    - model - DirichletMixtureModel object
    - n_inner - Number of inner iterations
    """
    def __init__(self, hp : PointProcessStorage, 
                    model : DirichletMixtureModel,
                    n_inner : int = 10):
        self.N = hp.N
        self.K = model.K
        self.C = model.mu.shape[0]
        self.hp = hp
        self.model = model
        self.n_inner = n_inner

        self.g = []
        self.int_g = []

        self.gg = []

    def learn_hp(self, niter=100):
        nll_history = []
        r = torch.rand(self.N, self.K)
        r = r / r.sum(1)[:, None]
        for _ in trange(niter):
            mu, A = self.m_step(r, self.n_inner)
            nll1 = self.hp_nll(r.sum(0) / self.N, mu, A)

            r2 = self.e_step()
            mu2, A2 = self.m_step(r2, self.n_inner)
            nll2 = self.hp_nll(r2.sum(0) / self.N, mu2, A2)

            if nll1 < nll2:
                nll_history.append(nll1.item())
                self.update_model(r, mu, A)
            else:
                nll_history.append(nll2.item())
                r = r2
                self.update_model(r, mu2, A2)

        return r, nll_history

    def update_model(self, r, mu, A):
        Sigma = A
        B = (2 / np.pi)**.5 * mu
        self.model.update_A(A, Sigma)
        self.model.update_mu(mu, B)
        pi = r.sum(0) / self.N
        self.model.update_pi(pi)

    def hp_nll(self, pi, mu, A):
        """
        Computes negative log-likelihood given \pi, \mu, A up to a constant (!)
        """
        assert torch.isclose(pi.sum(0), torch.ones_like(pi.sum(0))), pi
        nll = 0

        for n, (c, _, Tn) in enumerate(self.hp):
            g = self._get_g(n)
            A_g = (A[c.tolist(), c.tolist(), :, :] * g[:, :, :, None]).sum(2) # L x L x K
            lamda = mu[c.tolist(), :] + A_g.sum(1)
            assert (lamda > 0).all()

            int_g = self._get_int_g(n)
            A_g = (A[:, c.tolist(), :, :] * int_g[None, :, :, None]).sum(2)# C x K
            int_lambda = Tn * mu + A_g.sum(1)
            #integral = self.integral_lambda(t, c, Tn, mu, A, int_g=self._get_int_g(n))

            ll_lower_bound = (pi * (torch.log(lamda).sum(0) - int_lambda.sum(0))).sum(0)
            #assert (ll_lower_bound < 1).all(), ll_lower_bound

            nll -= ll_lower_bound

        return nll

    def _get_g(self, n):
        t = self.hp.seqs[n][:, 0]
        if len(self.g) <= n:
            tau = torch.tril(t.unsqueeze(1).repeat(1, t.shape[0]) - t[None, :], diagonal=-1)
            assert (tau >= 0).all()
            g = torch.stack([f(tau) for f in self.hp.basis_fs], dim=-1)
            g = g * (tau > 0)[:, :, None]
            self.g.append(g)
        else:
            g = self.g[n]

        return g

    def _get_int_g(self, n):
        t = self.hp.seqs[n][:, 0]
        Tn = self.hp.Tns[n]
        if len(self.int_g) <= n:
            int_g = torch.stack([integral(f, torch.zeros_like(t), Tn - t) for f in self.hp.basis_fs], dim=-1)
            self.int_g.append(int_g)
        else:
            int_g = self.int_g[n]
        
        return int_g

    def e_step(self):
        log_rho = torch.zeros(self.N, self.K)
        elogpi = self.model.e_logpi()
        log_rho += elogpi[None, :]
        
        e_mu = self.model.e_mu() # C x K
        e_A = self.model.e_A() # C x C x D x K
        var_mu = self.model.var_mu()

        for n, (c, t, Tn) in enumerate(self.hp):
            g = self._get_g(n)
            e_A_g = (e_A[c.tolist(), c.tolist(), :, :] * g[:, :, :, None]).sum(2).permute(2, 0, 1) # K x L x L
            e_lambda = e_mu[c.tolist(), :].permute(1, 0) + e_A_g.sum(-1)
            var_lambda = var_mu[c.tolist(), :].permute(1, 0) + ((e_A_g)**2).sum(1)
            log_rho[n, :] += (torch.log(e_lambda) - var_lambda / (2*e_lambda**2)).sum(1)

            int_g = self._get_int_g(n)
            int_lambda = (Tn * e_mu + (e_A[:, c.tolist(), :, :] * int_g[None, :, :, None]).sum(2).sum(1)).sum(0)
            log_rho[n, :] -= int_lambda

        rho = F.softmax(log_rho, -1)
        r = rho / rho.sum(1)[:, None]
        assert torch.isclose(r.sum(1), torch.ones_like(r.sum(1))).all(), r.sum(1)

        return r

    def m_step(self, r, niter=8):
        mu = (np.pi / 2)**.5 * self.model.B
        beta = self.model.B
        A = self.model.Sigma
        Sigma = self.model.Sigma
        C = mu.shape[0]
        
        for _ in range(niter):
            b = torch.zeros(self.K)
            c = -1 * torch.ones(self.C, self.K)
            s = 0
            d = 0
            for n, (cs, _, Tn) in enumerate(self.hp):
                b += r[n] * Tn # K 

                g = self._get_g(n)
                A_g = (A[cs.tolist(), cs.tolist(), :, :] * g[:, :, :, None])#.sum(2) # L x L x D x K
                lamda = mu[cs.tolist(), :] + A_g.sum(2).sum(1) 
                pii = mu[cs.tolist(), :] / lamda # L x K
                
                #A_g_ = (A[cs.tolist(), cs.tolist(), :, :] * g[:, :, :, None]) # L x L x D x K
                pijd = A_g / lamda[:, None, None, :] # L x L x D x K
                assert (pijd <= 1).all(), (pijd.max(), lamda.min(), mu.min())
                
                x = cs.unsqueeze(0).repeat(C, 1)
                eq = x == torch.tensor(np.arange(C)).unsqueeze(1).repeat(1, cs.shape[0]) # C x L 
                sum_pii = torch.stack([pii[torch.BoolTensor(mask)].sum(0) for mask in eq], dim=0) #None, :, :] * eq[:, :, None]).sum(1) # C x K
                c -= r[n][None, :] * sum_pii # C x K
                
                #sum_pijd = (pijd[None, :, :, :, :] * eq[:, :, None, None, None]).sum(1) # C L D K
                #sum_pijd = (sum_pijd[:, None, :, :, :] * eq[None, :, :, None, None]).sum(2) # C C D K
                sum_pijd = torch.stack([(pijd[torch.BoolTensor(mask)]).sum(0) for mask in eq], dim=0)
                sum_pijd = torch.stack([(sum_pijd[:, torch.BoolTensor(mask)]).sum(1) for mask in eq], dim=0)
                assert (sum_pijd >= 0).all()
                s += r[n][None, None, None, :] * sum_pijd # C C D K

                int_g = self._get_int_g(n)
                sum_int = (int_g[:, None, :] * eq.permute(1, 0)[:, :, None]).sum(0) # C D
                d += r[n][None, None, :] * sum_int[:, :, None] # C D K

            a = 1 / beta**2 # C x K 
            mu = 0.001 + (-b[None, :] + (b[None, :]**2 - 4*a*c)**.5)/ (2*a)
            A = s / (Sigma**(-1) + d[None, :, :, :])
            assert (A >= 0).all()
            assert (mu > 0).all()

        return mu, A


# class HawkesProcess:
#     """
#     Implementation of Hawkes process model
#     """
#     def __init__(self, mu, a, basis_fs, Ts, s):
#         """
#         """
#         self.mu = mu
#         self.a = a
#         self.basis_fs = basis_fs
#         self.Ts = Ts
#         self.s = s

#         self.N = len(s)
#         self.C = mu.shape[0]
#         self.lambdas = None

#     def intensity(self, ts, cs=None, s_ids=None):
#         s_ids = np.arange(self.N) if s_ids is None else s_ids
#         if cs is not None and len(cs) == len(ts):
#             lambdat = torch.zeros(len(s_ids), len(ts))
#         else:
#             lambdat = torch.zeros(len(s_ids), len(ts), self.C)
        
#         for n, sn in enumerate(self.s):
#             if n not in s_ids:
#                 continue
#             n_ = list(s_ids).index(n)

#             if cs is not None:
#                 lambdat[n_, :] = self.mu[cs.tolist()]
#             else:
#                 lambdat[n_, :, :] = self.mu[None, :]

#             basis_vals = list()
#             t_right_id = 0
#             t_left_id = 0
#             ts_id = 0
            
#             for t_id, ti in enumerate(sn[:, 0]):
#                 if ti >= ts[ts_id]:
#                     ts_id += 1
#                     if ts_id >= len(ts):
#                         break
#                     t_left_id  = t_right_id
#                     t_right_id = t_id
                
#                     if t_right_id - t_left_id > 0:
#                         t_res = ts[ts_id:].unsqueeze(1).repeat(1, t_right_id - t_left_id) - sn[t_left_id : t_right_id, 0].unsqueeze(0).repeat(len(ts) - ts_id, 1)
#                         cs_ = sn[t_left_id:t_right_id, 1]
#                         basis_vals = torch.stack([g(t_res) for g in self.basis_fs], dim=-1)
#                         if cs is not None:
#                             a_s = self.a[cs[ts_id:].tolist(), :, :][:, cs_.tolist(), :]
#                             lambdat[n_, ts_id:] += torch.einsum('abc,abc->a', a_s, torch.FloatTensor(basis_vals))
#                         else:
#                             a_s = self.a[:, cs_.tolist(), :]
#                             lambdat[n_, ts_id:, :] += torch.einsum('abc,dbc->da', a_s, torch.FloatTensor(basis_vals))#.sum(-1).sum(-1)

#         return lambdat

#     def intensity_in_real_points(self):
#         lambdas = []
#         for _, sn in enumerate(self.s):
#             #lamdan = self.mu
#             t_res = sn[:, 0].unsqueeze(0).repeat(len(sn[:, 0]), 1)
#             t_res = t_res - sn[:, 0][:, None]
#             cs = sn[:, 1].unsqueeze(0).repeat(len(sn[:, 1]), 1)
#             a_s = self.a[:, cs.tolist(), :]
#             basic_vals = torch.stack([g(t_res)*(t_res > 0).float() for g in self.basis_fs], dim=-1)
#             lambdan = self.mu[:, None ] + (a_s * basic_vals).sum(-1).sum(-1)
#             lambdas.append(lambdan)
#         return lambdas

#     def intensity_integral(self, n_pts=100):
#         ints = torch.zeros(self.N, self.C)
#         if isinstance(self.Ts, (float, int)):
#             ts = torch.FloatTensor(np.linspace(0, self.Ts, n_pts))
#             ints += (ts[1] - ts[0]) * self.intensity(ts).sum(1)
#         elif isinstance(self.Ts, list):
#             for n, T in enumerate(self.Ts):
#                 ts = torch.FloatTensor(np.linspace(0, T, n_pts))
#                 s_ids = [n]
#                 ints[n] += (ts[1] - ts[0]) * self.intensity(ts, s_ids=s_ids).sum(1)[0]
#         else:
#             raise ValueError()

#         return ints

#     @staticmethod
#     def tailor_exp_log_lambda(lambdas):
#         log_expectation = torch.log(lambdas).mean(0)
#         expectation2 = lambdas.mean(0)**2
#         var = (lambdas**2).mean(0) - lambdas.mean(0)**2
#         return log_expectation - var * (2 * expectation2)**(-1)

#     def log_likelihood(self):
#         log_likelihood = 0
#         lambdas = [self.intensity(sn[:, 0], cs=sn[:, 1]) for sn in self.s]
#         log_likelihood += torch.stack([HawkesProcess.tailor_exp_log_lambda(l).sum(0) for l in lambdas], dim=0)
#         log_likelihood -= self.intensity_integral().sum(-1)
#         return log_likelihood
        
