import torch
import numpy as np

class HawkesProcess:
    """
    Implementation of Hawkes process model
    """
    def __init__(self, mu, a, basis_fs, T, s):
        """
        """
        self.mu = mu
        self.a = a
        self.basis_fs = basis_fs
        self.T = T
        self.s = s

        self.N = len(s)
        self.C = mu.shape[0]
        self.lambdas = None

    def intensity(self, t, s_ids=None): 
        lambdat = torch.zeros(self.N, self.C)
        for n, sn in enumerate(self.s):
            basis_vals = list()
            t_right_id = 0
            for t_id, ti in enumerate(sn[:, 0]):
                if ti >= t:
                    t_right_id = t_id

            if t_right_id == 0:
                continue
            t_res = t - sn[:t_right_id, 0]
            cs = sn[:t_right_id, 1]
            a_s = self.a[:, cs.tolist(), :]
            basis_vals = torch.stack([g(t_res) for g in self.basis_fs], dim=-1)
            lambdat[n] = self.mu + (a_s * torch.FloatTensor(basis_vals)[None, :, :]).sum(-1).sum(-1)
        
        return lambdat

    def intensity_in_real_points(self):
        lambdas = []
        for n, sn in enumerate(self.s):
            #lamdan = self.mu
            t_res = sn[:, 0].unsqueeze(0).repeat(len(sn[:, 0]), 1)
            t_res = t_res - sn[:, 0][:, None]
            cs = sn[:, 1].unsqueeze(0).repeat(len(sn[:, 1]), 1)
            a_s = self.a[:, cs.tolist(), :]
            basic_vals = torch.stack([g(t_res)*(t_res > 0).float() for g in self.basis_fs], dim=-1)
            lambdan = self.mu[:, None ] + (a_s * basic_vals).sum(-1).sum(-1)
            lambdas.append(lambdan)
        return lambdas

    def intensity_integral(self, n_pts=100):
        ints = torch.zeros(self.N, self.C)
        ts = np.linspace(0, self.T, n_pts)
        for t in ts:
            ints += (ts[1] - ts[0]) * self.intensity(t)
        return ints

    def log_likelihood(self):
        log_likelihood = 0
        lambdas = self.intensity_in_real_points()
        self.lambdas = lambdas
        log_likelihood = torch.log(torch.stack([l.sum() for l in lambdas], dim=0))
        log_likelihood -= self.intensity_integral().sum(-1)
        return log_likelihood
        
        
class EM_clustering:
    def __init__(self, hps):
        pass

    def e_step(self):
        raise NotImplementedError()

    def m_step(self):
        raise NotImplementedError()
