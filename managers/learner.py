from cgitb import text
from turtle import forward
import torch
import multiprocessing as mp

from torch.utils.data import DataLoader, RandomSampler
from utils.utils import *
from gt_utils.torch_utils import collate_func, move_batch_to_device
from sklearn import metrics as met

class Learner():
    def __init__(self, name, data, model, metrics, optimizer):
        self.name = name
        self.data = data
        self.model = model
        self.metrics = metrics
        self.optimizer_type = optimizer
        self.loss = torch.nn.MSELoss()

        self.current_dataloader = None
        self.optimizer = None

    def create_dataloader(self, batch_size, num_workers=4):
        self.current_dataloader = DataLoader(self.data, batch_size=batch_size, num_workers=num_workers,  shuffle=True, pin_memory=True)
        return self.current_dataloader

    def initialize_metrics(self):
        for k in self.metrics:
            self.metrics[k] = 0

    def preprocess(self, device=None):
        pass

    def load(self, batch, device):
        batch[0] = batch[0].float()
        batch[1] = batch[1].float()
        batch[1] = batch[1].view(-1,1)
        return batch

    def forward_func(self, batch):
        input, _ = batch
        return self.model(input)

    def loss_func(self, res, batch):
        return self.loss(res, batch[1])

    def score_func(self, loss, res, batch):
        self.metrics['loss']+=loss

    def epoch_results(self):
        metrics = {}
        for k in self.metrics:
            metrics[k] = self.metrics[k]/len(self.data)
        return metrics

    def setup_optimizer(self, optimizer_groups):
        parameters = [p for p in self.model.parameters()]
        optimizer_groups[0]['params'] = parameters
        self.optimizer = self.optimizer_type(optimizer_groups)

    def save_model(self, path, epoch):
        torch.save({'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, path)

    def load_model(self, path, device=None):
        state_d = torch.load(path, device)
        self.optimizer.load_state_dict(state_d['optimizer'])
        self.model.load_state_dict(state_d['state_dict'])
        return state_d['epoch']


class CCALearner(Learner):
    def __init__(self, name, data, model, metrics, optimizer, reg_model=None):
        super().__init__(name, data, model, metrics, optimizer)
        self.reg_model = reg_model

    def create_dataloader(self, batch_size=4096, num_workers=4, dry_run=False):
        self.current_dataloader = DataLoader(self.data, batch_size=batch_size, num_workers=num_workers,  shuffle=True, pin_memory=True, collate_fn=collate_func)
        return self.current_dataloader
    
    def preprocess(self, device=None):
        self.model.train()
        self.model.use_precomputed_emb = False
        self.model.batch_project = True

    def setup_optimizer(self, optimizer_groups):
        # optimizer_groups[0]['params'] = [p for p in self.model.parameters()]
        op_params = [p for p in self.model.parameters()]
        optimizer_groups[0]['params'] = op_params
        self.optimizer = self.optimizer_type(optimizer_groups)
        
    
    def load(self, batch, device):
        batch = batch.ls
        batch = move_batch_to_device(batch, device)
        return batch
    
    def forward_func(self, batch):
        node_id, context, ane_type, los = batch
        g = self.data.g.to(node_id.device)
        res = self.model(g, (context, node_id))
        return res

    def loss_func(self, res, batch):
        loss = res[0]
        return loss

    def score_func(self, loss, res, batch):
        self.metrics['total_cor'] += res[0].item()
        self.metrics['total_graph_var'] += res[1].item()
        self.metrics['total_ft_var'] += res[2].item()

    def epoch_results(self):
        metrics = {}
        for k in self.metrics:
            metrics[k] = self.metrics[k]/len(self.current_dataloader)
        return metrics
    
class RegCotrainLearner(Learner):
    def __init__(self, name, data, model, metrics, optimizer, reg_model=None):
        self.processed_graph = None
        self.reg_model = reg_model
        super().__init__(name, data, model, metrics, optimizer)

    def create_dataloader(self, batch_size=64, num_workers=4, dry_run=False):
        self.current_dataloader = DataLoader(self.data, batch_size=batch_size, num_workers=num_workers,  shuffle=True, pin_memory=True, collate_fn=collate_func)
        return self.current_dataloader
    
    def preprocess(self, device=None):
        self.model.eval()
        for m in self.reg_model:
            m.train()
        self.model.batch_project = False
        self.model.use_precomputed_emb = True
        self.processed_graph = self.data.g.to(device)
        with torch.no_grad():
            self.model.graph_update(self.processed_graph)
        return None

    def setup_optimizer(self, optimizer_groups):
        op_params = []
        if len(optimizer_groups)==1:
            for i, m in enumerate(self.reg_model):
                op_params += [p for p in m.parameters()]
            optimizer_groups[0]['params'] = op_params
        else:
            for i, m in enumerate(self.reg_model):
                optimizer_groups[i]['params'] = [p for p in m.parameters()]
        self.optimizer = self.optimizer_type(optimizer_groups)
    
    def load(self, batch, device):
        batch = batch.ls
        batch = move_batch_to_device(batch, device)
        return batch
    
    def forward_func(self, batch):
        node_id, context, ane_type, los = batch
        score, graph_var, ft_var, gn_repr, ft_repr = self.model(self.processed_graph, (context, node_id))
        g_pred = self.reg_model[0](gn_repr)
        f_pred = self.reg_model[1](ft_repr)
        return g_pred, f_pred

    def loss_func(self, res, batch):
        loss1 = self.loss(res[0], batch[3].view(-1,1))
        loss2 = self.loss(res[1], batch[3].view(-1,1))
        return loss1+loss2

    def score_func(self, loss, res, batch):
        self.metrics['mse1'] += self.loss(res[0], batch[3].view(-1,1)).item()
        self.metrics['mse2'] += self.loss(res[1], batch[3].view(-1,1)).item()

    def epoch_results(self):
        metrics = {}
        for k in self.metrics:
            metrics[k] = self.metrics[k]/len(self.current_dataloader)
        return metrics
    
    def load_model(self, path, device=None):
        state_d = torch.load(path, device)
        self.model.load_state_dict(state_d['state_dict'])
        return state_d['epoch']

class GraphCotrainLearner(Learner):
    def __init__(self, name, data, model, metrics, optimizer, reg_model=None):
        self.reg_model = reg_model
        self.processed_graph = None
        super().__init__(name, data, model, metrics, optimizer)

    def create_dataloader(self, batch_size=4096, num_workers=4, dry_run=False):
        self.current_dataloader = DataLoader(self.data, batch_size=batch_size, num_workers=num_workers,  shuffle=True, pin_memory=True, collate_fn=collate_func)
        return self.current_dataloader
    
    def preprocess(self, device=None):
        self.model.eval()
        for r in self.reg_model:
            r.eval()
        self.processed_graph = self.data.g.to(device)
        self.model.graph_update(self.processed_graph)
        self.model.batch_project = False
        self.model.use_precomputed_emb = True
    
    def load(self, batch, device):
        batch = batch.ls
        batch = move_batch_to_device(batch, device)
        return batch
    
    def forward_func(self, batch):
        node_id, context, ane_type, los = batch
        score, graph_var, ft_var, gn_repr, ft_repr = self.model(self.processed_graph, (context, node_id))
        g_pred = self.reg_model[0](gn_repr)
        f_pred = self.reg_model[1](ft_repr)
        return score,g_pred, f_pred, graph_var, ft_var

    def score_func(self, loss, res, batch):
        self.metrics['total_cor'] += res[0].item()
        self.metrics['total_graph_var'] += res[3].item()
        self.metrics['total_ft_var'] += res[4].item()
        self.metrics['mse1'] += self.loss(res[1], batch[3].view(-1,1)).item()
        self.metrics['mse2'] += self.loss(res[2], batch[3].view(-1,1)).item()

    def epoch_results(self):
        metrics = {}
        for k in self.metrics:
            metrics[k] = self.metrics[k]/len(self.current_dataloader)
        return metrics

    def loss_func(self, res, batch):
        return None

class ProjectionLearner(Learner):
    def __init__(self, name, data, model, metrics, optimizer, reg_model=None):
        self.processed_graph = None
        self.gn_repr_list = []
        self.ft_repr_list = []
        super().__init__(name, data, model, metrics, optimizer)

    def initialize_metrics(self):
        super().initialize_metrics()
        self.gn_repr_list = []
        self.ft_repr_list = []

    def create_dataloader(self, batch_size=4096, num_workers=4, dry_run=False):
        self.current_dataloader = DataLoader(self.data, batch_size=batch_size, num_workers=num_workers,  shuffle=True, pin_memory=True, collate_fn=collate_func)
        return self.current_dataloader
    
    def preprocess(self, device=None):
        self.model.eval()
        self.processed_graph = self.data.g.to(device)
        self.model.graph_update(self.processed_graph)

    def loss_func(self, res, batch):
        return None
    
    def load(self, batch, device):
        batch = batch.ls
        batch = move_batch_to_device(batch, device)
        return batch
    
    def forward_func(self, batch):
        node_id, context, ane_type, los = batch
        gn_repr = self.model.get_graph_embedding(self.processed_graph, node_id)
        ft_repr = self.model.text_update(context)
        return gn_repr, ft_repr

    def score_func(self, loss, res, batch):
        self.gn_repr_list.append(res[0])
        self.ft_repr_list.append(res[1])

    def epoch_results(self):
        metrics = {}
        for k in self.metrics:
            metrics[k] = self.metrics[k]/len(self.current_dataloader)
        graph_emb = torch.cat(self.gn_repr_list, dim=0)
        text_embs = torch.cat(self.ft_repr_list, dim=0)
        corr, U, V = self.model.corr_eval.loss(graph_emb, text_embs)
        metrics['correlation'] = corr
        self.model.U = U
        self.model.V = V
        return metrics