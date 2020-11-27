import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

class Soft_DTW(nn.Module):
    def __init__(self, gamma=1.0, norm=False):
        super(Soft_DTW, self).__init__()
        self.norm = norm
        self.gamma = gamma
        self.dtw = Func_SDTW.apply

    def dist_matrix(self, x, y):
        dim0, dim1, dim2 = x.size(1), y.size(1), x.size(2)
        x = x.unsqueeze(2).expand(-1, dim0, dim1, dim2)
        y = y.unsqueeze(1).expand(-1, dim0, dim1, dim2)

        return torch.pow(x - y, 2).sum(3)

    def forward(self, x, y):

        # preparations
        assert len(x.shape) == len(y.shape)
        unsqueezed = False
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            unsqueezed = True
        
        # find distance matrix between x, y
        D_xy = self.dist_matrix(x, y)
        
        # calc soft-DTW on martix D
        out_xy = self.dtw(D_xy, self.gamma)
        
        # normalize if needed
        if self.norm:
            D_xx = self.dist_matrix(x, x)
            out_xx = self.dtw(D_xx, self.gamma)
            D_yy = self.dist_matrix(y, y)
            out_yy = self.dtw(D_yy, self.gamma)
            output = out_xy - (out_xx + out_yy)/2
        else:
            output = out_xy
        
        return output.squeeze(0) if unsqueezed else output

class Func_SDTW(Function):
    def forward(ctx, D, gamma):
        R = torch.Tensor(sdtw_forward(D, gamma)).to(D.device)
        ctx.save_for_backward(D, R, gamma)
        
        return R[:, -2, -2]

    def backward(ctx, grad_output):
        D, R, gamma = ctx.saved_tensors
        E = torch.Tensor(sdtw_backward(D, R, gamma.item())).to(grad_output.device)
        
        return grad_output.view(-1, 1, 1).expand_as(E)*E, None

def sdtw_forward(D, gamma):
    dim0, dim1, dim2 = D.shape
    R = torch.ones((dim0, dim1+2, dim2+2))*np.inf
    R[:, 0, 0] = 0
    
    for i in range(dim0):
        for j in range(1, dim2+1):
            for k in range(1, dim1+1):
                z0 = -R[i, k-1, j-1]/gamma
                z1 = -R[i, k-1, j]/gamma
                z2 = -R[i, k, j-1]/gamma
                zmax = max(z0, z1, z2)
                zsum = torch.exp(z0 - zmax) + torch.exp(z1 - zmax) + torch.exp(z2 - zmax)
                softmin = -(torch.log(zsum) + zmax)*gamma
                R[i, k, j] = D[i, k-1, j-1] + softmin      
    return R

def sdtw_backward(D, R, gamma):
    dim0, dim1, dim2 = D.size()
    D1 = E = torch.zeros((dim0, dim1+2, dim2+2))
    D1[:, 1:N+1, 1:M+1] = D
    E[:, -1, -1] = 1
    R[:, : , -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    
    for i in range(dim0):
        for j in range(dim2, 0, -1):
            for k in range(dim1, 0, -1):
                a = torch.exp((R[i, k+1, j] - R[i, k, j] - D1[i, k+1, j])/gamma)
                b = torch.exp((R[i, k, j+1] - R[i, k, j] - D1[i, k, j+1])/gamma)
                c = torch.exp((R[i, k+1, j+1] - R[i, k, j] - D1[i, k+1, j+1])/gamma)
                E[i, k, j] = E[i, k+1, j]*a + E[i, k, j+1]*b + E[i, k+1, j+1]*c
    
    return E[:, 1:N+1, 1:M+1]