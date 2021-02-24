import numpy as np
import torch

from DMHP.metrics import purity
from src.optimal_transport.sk import sinkhorn, get_labels


class Trainer:
    def __init__(self, model, optimizer, criterion, device,\
                 X, n_classes, gt = None, max_epochs = 200, lamb = 10, nopts = 400, batch_size = 30,\
                lr = 0.0005, alr = 0.0005):
        self.N = len(X[0])
        self.optimize_times = ((max_epochs + 1.0001)*self.N*(np.linspace(0, 1, nopts))[::-1]).tolist()
        self.optimize_times = [(max_epochs +10)*self.N] + self.optimize_times
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.X = X
        self.gt = gt
        self.max_epochs = max_epochs
        self. lamb = lamb
        self.batch_size = batch_size
        self.lr = lr
        self.alr = alr
        self.K = n_classes
        self.selflabels = np.random.permutation(list(range(n_classes))*(self.N//n_classes))
        if len(self.selflabels)!=self.N:
            self.selflabels = np.concatenate((self.selflabels, np.zeros(self.N - len(self.selflabels))), axis = None)
    
    def adjust_learning_rate(self,epoch):
        lr = self.alr
        if self.max_epochs == 200:
            if epoch >= 20:
                lr = self.alr * (0.1 ** ((epoch - 20) // 10))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif self.max_epochs == 400:
            if epoch >= 160:
                lr = self.alr * (0.1 ** ((epoch - 160) // 80))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif self.max_epochs == 800:
            if epoch >= 320:
                lr = self.alr * (0.1 ** ((epoch - 320) // 160))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif self.max_epochs == 1600:
            if epoch >= 640:
                lr = self.alr * (0.1 ** ((epoch - 640) // 320))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
    def train_epoch(self, epoch):
        self.adjust_learning_rate(epoch)
        indices = np.random.permutation(self.N)
        self.model.train()
        
        losses = []
        for iteration, start in enumerate(range(0, self.N - self.batch_size, self.batch_size)):
            batch_ids = indices[start:start+self.batch_size]
            batch = [j[batch_ids].to(self.device) for j in self.X]
            
            n_iteration = epoch*self.N//self.batch_size + iteration
            if n_iteration*self.batch_size >= self.optimize_times[-1]:
                with torch.no_grad():
                    _ = self.optimize_times.pop()
                    self.selflabels = get_labels(sinkhorn(torch.log(torch.softmax(self.model(self.X), dim = 0)).T/self.N,\
                                                          torch.ones(self.K)/self.K,\
                                                          torch.ones(self.N)/self.N, eps=1e-3))
            self.optimizer.zero_grad()
            out = self.model(batch)
            loss = self.criterion(out, self.selflabels[batch_ids])
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
        print ('loss:', np.mean(losses))
    
    def train(self):
        if self.gt is not None:
            max_purity = 0
        else:
            max_purity = None
        best_sl = None
        sls = []
        for epoch in range(self.max_epochs):
            self.train_epoch(epoch)
            self.model.eval()
            sl = get_labels(self.model(self.X).T)
            sls.append(sl)
            if self.gt is not None:
                res = purity(sl, self.gt)
                if res > max_purity:
                    max_purity = res
                    best_sl = sl
#                 print("On epoch {} purity: {}".format(epoch, res))
        return best_sl, max_purity, sls
