import torch
from torch_utils import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score
import time

class Learner():
    def __init__(self):
        pass

    def data_func():
        raise NotImplementedError
    
    def preprocess():
        raise NotImplementedError

    def forward_func():
        raise NotImplementedError

    def get_dataloader():
        raise NotImplementedError
    
    def loss_func():
        raise NotImplementedError

    def score_func():
        raise NotImplementedError

class CorrelationLearner(Learner):
    def __init__(self, model):
        self.metrics = {
            'total_cor':0,
            'total_graph_var':0,
            'total_ft_var':0,
            'total_loss':0
        }
        self.model = model

    def data_func(self, data):
        return (data[1], data[0])

    def preprocess(self, graph):
        self.model.train()
        return None

    def forward_func(self, g, data):
        self.model(g, data)
    
    def get_dataloader(data):
        DataLoader(data, batch_size=512, num_workers=4,  shuffle=True, collate_fn=collate_func)
    
    def loss_func(self, res, data):
        loss = res[0]
        return loss

    def score_func(self, loss, res, data):
        self.metrics['total_loss'] += res[0].item()
        smetrics['total_cor'] += res[0].item()
        metrics['total_graph_var'] += res[1].item()
        metrics['total_ft_var'] += res[2].item()