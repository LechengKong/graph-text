from torch.utils.data import Dataset, DataLoader

class NodeTextDataset(Dataset):
    def __init__(self, data, graph):
        self.data = data
        self.num_sample = len(data)
        self.g = graph

    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        node_id, context, context_len, code_desc, ct_len, ane_type, los = self.data[index]
        return node_id, context, context_len, code_desc, ct_len, int(ane_type), los