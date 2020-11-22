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

    def intensity(self, ts, cs=None, s_ids=None):
        s_ids = np.arange(self.N) if s_ids is None else s_ids
        if cs is not None and len(cs) == len(ts):
            lambdat = torch.zeros(len(s_ids), len(ts))
        else:
            lambdat = torch.zeros(len(s_ids), len(ts), self.C)
        
        for n, sn in enumerate(self.s):
            if n not in s_ids:
                continue
            n_ = list(s_ids).index(n)

            if cs is not None:
                lambdat[n_, :] = self.mu[cs.tolist()]
            else:
                lambdat[n_, :, :] = self.mu[None, :]

            basis_vals = list()
            t_right_id = 0
            t_left_id = 0
            ts_id = 0
            
            for t_id, ti in enumerate(sn[:, 0]):
                if ti >= ts[ts_id]:
                    ts_id += 1
                    if ts_id >= len(ts):
                        break
                    t_left_id  = t_right_id
                    t_right_id = t_id
                
                    if t_right_id - t_left_id > 0:
                        t_res = ts[ts_id:].unsqueeze(1).repeat(1, t_right_id - t_left_id) - sn[t_left_id : t_right_id, 0].unsqueeze(0).repeat(len(ts) - ts_id, 1)
                        cs_ = sn[t_left_id:t_right_id, 1]
                        basis_vals = torch.stack([g(t_res) for g in self.basis_fs], dim=-1)
                        if cs is not None:
                            a_s = self.a[cs[ts_id:].tolist(), :, :][:, cs_.tolist(), :]
                            lambdat[n_, ts_id:] += torch.einsum('abc,abc->a', a_s, torch.FloatTensor(basis_vals))
                        else:
                            a_s = self.a[:, cs_.tolist(), :]
                            lambdat[n_, ts_id:, :] += torch.einsum('abc,dbc->da', a_s, torch.FloatTensor(basis_vals))#.sum(-1).sum(-1)

        return lambdat

    def intensity_in_real_points(self):
        lambdas = []
        for _, sn in enumerate(self.s):
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
        if isinstance(self.Ts, (float, int)):
            ts = torch.FloatTensor(np.linspace(0, self.Ts, n_pts))
            ints += (ts[1] - ts[0]) * self.intensity(ts).sum(1)
        elif isinstance(self.Ts, list):
            for n, T in enumerate(self.Ts):
                ts = torch.FloatTensor(np.linspace(0, T, n_pts))
                s_ids = [n]
                ints[n] += (ts[1] - ts[0]) * self.intensity(ts, s_ids=s_ids).sum(1)[0]
        else:
            raise ValueError()

        return ints

    @staticmethod
    def tailor_exp_log_lambda(lambdas):
        log_expectation = torch.log(lambdas).mean(0)
        expectation2 = lambdas.mean(0)**2
        var = (lambdas**2).mean(0) - lambdas.mean(0)**2
        return log_expectation - var * (2 * expectation2)**(-1)


    def log_likelihood(self):
        log_likelihood = 0
        lambdas = [self.intensity(sn[:, 0], cs=sn[:, 1]) for sn in self.s]
        log_likelihood += torch.stack([HawkesProcess.tailor_exp_log_lambda(l).sum(0) for l in lambdas], dim=0)
        log_likelihood -= self.intensity_integral().sum(-1)
        return log_likelihood


class DirichletMixtureModel():
    def __init__(self, K, C, D, alpha, B=None, Sigma=None):
        self.alpha = alpha'
        self.K = K
        concentration = torch.FloatTensor([alpha /  float(K) for _ in range(K)])
        self.dirich = torch.distributions.dirichlet.Dirichlet(concentration)
        self.pi = self.dirich.sample()
        self.cat = torch.distributions.categorical.Categorical(self.pi)
        self.k_pi = self.cat.sample()
        self.rayleigh = torch.distributions.weibull.Weibull(2**0.5 * B, 2 * torch.ones_like(B))
        self.mu = self.rayleigh.sample()

        self.exp = torch.distributions.exponential.Exponential((Sigma)**(-1))
        self.A = self.exp.sample()

    def p_mu(self, mu=None):
        mu = mu if mu is not None else self.mu
        return torch.exp(self.rayleigh.log_prob(mu))

    def p_A(self, A=None):
        A = A if A is not None else self.A
        return torch.exp(self.exp.log_prob(A))

    def p_pi(self, pi=None):
        pi = pi if pi is not None else self.pi
        return torch.exp(self.dirich.log_prob(pi))

    def expect_log_pi(self, n_pts=100):
        return torch.digamma(self.alpha / self.K) - torch.digamma(self.alpha)
        # expect = 0
        # for _ in range(n_pts):
        #     log_prob = self.dirich.log_prob(self.dirich.sample())
        #     expect += log_prob * torch.exp(log_prob)
        # return expect

        
class EM_clustering:
    def __init__(self, hps, Z, model):
        self.Z = Z
        self.hps = hps
        self.model = model
        self.K = len(hps)
        self.N = len(self.hps[0].s)

    def e_step(self):
        log_q = 0
        e_log_pi = self.model.expect_log_pi()
        rho = torch.zeros(self.N, self.K)
        for k in range(self.K):
            #zk = Z[:, k]
            rho[:, k] = self.hps[k].log_likelihood() + \
                self.model.e_log_pi
        respon = rho / rho.sum(1)
        return respon 
            
    def m_step(self):

        p_mu = self.model.p_mu()
        p_A = self.model.p_A()

        
    #         % E-step: evaluate the responsibility using the current parameters
    #     for c = 1:length(Seqs)

    #         Time = Seqs(c).Time;
    #         Event = Seqs(c).Mark;
    #         Tstart = Seqs(c).Start;
                
    #         if isempty(alg.Tmax)
    #             Tstop = Seqs(c).Stop;
    #         else
    #             Tstop = alg.Tmax;
    #             indt = Time < alg.Tmax;
    #             Time = Time(indt);
    #             Event = Event(indt);
    #         end

    #         N = length(Time);
    #         % calculate the integral decay function in the log-likelihood function
    #         G = Kernel_Integration(Tstop - Time, model);


    #         TMPAA = zeros(size(A));
    #         TMPAB = zeros(size(A));
    #         %TMPMuB = zeros(size(mu));
    #         TMPMuC = zeros(size(mu));
    #         LL = 0;
                
    #         for i = 1:N

    #             ui = Event(i);
    #             ti = Time(i);
    #             TMPAA(ui,:,:,:) = TMPAA(ui,:,:,:)+ ...
    #                         repmat(G(i,:), [1, 1, model.K, model.D]);


    #             lambdai = mu(ui,:)+eps;
    #             pii = lambdai;
                    
    #             if i>1
    #                 tj = Time(1:i-1);
    #                 uj = Event(1:i-1);

    #                 gij = Kernel(ti-tj, model);
    #                 auiuj = A(uj, :, :, ui);
    #                 pij = repmat(gij, [1,1,model.K,1]).* auiuj;
                        
                        
    #                 tmp = sum(sum(pij,1),2);
    #                 lambdai = lambdai + tmp(:)';
                        
                        
    #                 pij = pij./repmat(reshape(lambdai,[1,1,model.K]),...
    #                         [size(pij,1),size(pij,2), 1]);
                        
    #                 for j=1:i-1
    #                     uj = Event(j);
    #                     TMPAB(uj,:,:,ui) = TMPAB(uj,:,:,ui) - pij(j,:,:);
    #                 end
                    
    #             end

    #             LL = LL+log(lambdai);
                    
                    
                    
    #             pii = pii./lambdai;                
    #             TMPMuC(ui,:)=TMPMuC(ui,:)-pii;
                    
    #         end
    #         LL = LL - (Tstop-Tstart).*sum(mu);
    #         tmp = sum(sum(repmat(G, [1,1,model.K]).*sum(A(Event,:,:,:),4),2),1);
    #         LL = LL - tmp(:)';

    #         % XX = (LL - max(LL));
    #         % EX(c,:)=(model.p'.*(exp(XX)+options.bias))./((exp(XX)+options.bias)*model.p);
                
    #         MuB = MuB + (Tstop-Tstart)*EX(c,:);
    #         for k=1:model.K
    #             AA(:,:,k,:) = AA(:,:,k,:) + EX(c,k)*TMPAA(:,:,k,:);
    #             AB(:,:,k,:) = AB(:,:,k,:) + EX(c,k)*TMPAB(:,:,k,:);
    #             MuC(:,k) = MuC(:,k) + EX(c,k)*TMPMuC(:,k);                
    #         end
    #         NLL = NLL - EX(c,:)*LL(:);
                
    #     end
            
    #     MuBB = repmat(MuB,[model.D, 1]);    
    #     % M-step: update parameters
    #     mutmp = (-MuBB+sqrt(MuBB.^2 - 4*MuA.*MuC))./(2*MuA);
    #     Atmp = -AB./AA;  
            
            
    #     Atmp(isnan(Atmp)) = 0;
    #     Atmp(isinf(Atmp)) = 0;
    #     mutmp(isnan(mutmp)) = 0;
    #     mutmp(isinf(mutmp)) = 0;
            
    #     % check convergence
    #     Err=sum(abs(A(:)-Atmp(:)))/sum(abs(A(:)));
    #     fprintf('Inner=%d, Obj=%f, RelErr=%f, Time=%0.2fsec\n',...
    #             in, NLL, Err, toc);

    #     A = Atmp;
    #     mu = mutmp;
    #     if Err<alg.thres || in==alg.inner
    #         break;
    #     end    
        
    # end    

    # model.beta = A;
    # model.b = sqrt(2/pi)*mu;

        
