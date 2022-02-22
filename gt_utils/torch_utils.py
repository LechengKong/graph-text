import torch
import numpy as np

class SCBatch:
    def __init__(self, samples):
        d = zip(*samples)
        node_id, context, ane_type, los = map(np.array, d)
        node_id = torch.tensor(node_id)
        context = torch.tensor(context)
        ane_type = torch.tensor(ane_type, dtype=torch.long)
        los = torch.tensor(los, dtype=torch.float)
        self.ls = [node_id, context, ane_type, los]

    def pin_memory(self):
        for i in range(len(self.ls)):
            self.ls[i] = self.ls[i].pin_memory()
        return self


def collate_func(samples):
    return SCBatch(samples)

def move_batch_to_device(batch, device):
    d = []
    for g in batch:
        d.append(g.to(device=device))
    return d