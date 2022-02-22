import sys
import torch
import dgl
import numpy as np
import torch
import random
import time

class SCBatch:
    def __init__(self, samples):
        d = zip(*samples)
        self.ls = []
        f = True
        for d1 in d:
            if f:
                b_l = [len(l) for l in d1]
                b_l = torch.tensor(b_l,dtype=torch.long)
                f= False
            p = np.concatenate(d1)
            p = torch.tensor(p,dtype=torch.long)
            self.ls.append(p)
        self.ls.append(b_l)

    def pin_memory(self):
        for i in range(len(self.ls)):
            self.ls[i] = self.ls[i].pin_memory()
        return self


def collate_dgl_onlylink(samples):
    return SCBatch(samples)

def move_batch_to_device(batch,device):
    d = []
    for g in batch:
        d.append(g.to(device=device))

    return d