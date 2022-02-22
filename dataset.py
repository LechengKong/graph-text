from torch.utils.data import Dataset, DataLoader
import math

class NodeTextDataset(Dataset):
    def __init__(self, data, graph):
        self.data = data
        self.num_sample = len(data)
        self.g = graph

    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        node_id, context, ane_type, los = self.data[index]
        return node_id, context, int(ane_type), math.log(los+1)