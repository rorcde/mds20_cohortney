import torch
import numpy as np


def consistency(trials_labels):
    J = len(trials_labels)
    values = torch.zeros(J)
    
    for trial_id, labels in enumerate(trials_labels):
        for trial_id2, labels2 in enumerate(trials_labels):
            if trial_id == trial_id2:
                continue
            ks = torch.unique(labels)
            sz_M = 0
            for k in ks:
                mask = labels == k
                s = mask.sum() * (mask.sum() - 1.) / 2.
                sz_M += s

            for k in ks:
                mask = labels == k
                s2 = 0
                for k2 in labels2[mask].unique():
                    sz_ = (labels2[mask] == k2).sum()
                    s2 += sz_ * (sz_ - 1.) / 2.
                values[trial_id] += (s - s2) / ((J-1) * sz_M)
    
    return torch.min(values)


