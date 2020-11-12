import torch
import numpy as np

class HawkesProcess:
    """
    Implementation of Hawkes process model
    """
    def __init__(self, mu, a, basis_fs, Ts, s):
        """
        """
        self.mu = mu
        self.a = a
        self.basis_fs = basis_fs
        self.Ts = Ts
        self.s = s

        self.N = len(s)
        self.C = mu.shape[0]
        self.lambdas = None

    def intensity(self, t, s_ids=None):
        s_ids = np.arange(self.N) if s_ids is None else s_ids
        lambdat = [] #torch.zeros(len(s_ids), self.C)

        for n, sn in enumerate(self.s):
            if n not in s_ids:
                continue
            basis_vals = list()
            t_right_id = 0
            for t_id, ti in enumerate(sn[:, 0]):
                if ti >= t:
                    t_right_id = t_id

            if t_right_id == 0:
                lambdat.append(self.mu)
                continue
            t_res = t - sn[:t_right_id, 0]
            cs = sn[:t_right_id, 1]
            a_s = self.a[:, cs.tolist(), :]
            basis_vals = torch.stack([g(t_res) for g in self.basis_fs], dim=-1)
            lambdat.append(self.mu + (a_s * torch.FloatTensor(basis_vals)[None, :, :]).sum(-1).sum(-1))

        lambdat = torch.stack(lambdat, dim=0)
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
        if isinstance(self.Ts, int):
            ts = np.linspace(0, self.Ts, n_pts)
            for t in ts:
                ints += (ts[1] - ts[0]) * self.intensity(t)
        elif isinstance(self.Ts, list):
            for n, T in enumerate(self.Ts):
                ts = np.linspace(0, T, n_pts)
                s_ids = [n]
                for t in ts:
                    ints[n] += (ts[1] - ts[0]) * self.intensity(t, s_ids)[0, :]
        else:
            raise ValueError()

        return ints

    def log_likelihood(self):
        log_likelihood = 0
        lambdas = self.intensity_in_real_points()
        self.lambdas = lambdas
        log_likelihood = torch.log(torch.stack([l.sum() for l in lambdas], dim=0))
        log_likelihood -= self.intensity_integral().sum(-1)
        return log_likelihood
        
        
class EM_clustering:
    def __init__(self, ):
        pass

    def e_step(self):
        raise NotImplementedError()

    def m_step(self):
        raise NotImplementedError()
