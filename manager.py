import torch
from torch_utils import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, accuracy_score
import time

class Manager():
    def __init__(self, train_set, val_set, model, params, reg_model=None):
        self.train_set = train_set
        self.valid_set = val_set
        self.model = model
        self.reg_model = reg_model
        self.params = params
        self.start_ep = 0

    def setup(self, train_reg=False):
        if train_reg:
            op_param = []
            for i in range(len(self.reg_model)):
                op_param +=[p for p in self.reg_model[i].parameters()]
            self.optimizer = torch.optim.Adam(op_param, lr=self.params.lr)
            self.loss = torch.nn.MSELoss()
            state_d = torch.load(self.params.model_file_name, map_location=self.params.device)
            self.model.load_state_dict(state_d['state_dict'])
            self.model.eval()
            self.model.requires_grad = False
            print('model loaded')
        else:
            op_param = [p for p in self.model.parameters()]
            for i in range(len(self.reg_model)):
                op_param += [p for p in self.reg_model[i].parameters()]
            self.optimizer = torch.optim.Adam(op_param, lr=self.params.lr)
            self.cel = torch.nn.CrossEntropyLoss()
            self.loss = torch.nn.MSELoss()
            if self.params.retrain:
                state_d = torch.load(self.params.model_file_name, map_location=self.params.device)
                self.optimizer.load_state_dict(state_d['optimizer'])
                self.model.load_state_dict(state_d['state_dict'])
                self.start_ep = state_d['epoch']+1
    
    def train(self):
        # self.eval_epoch()
        for epoch in range(self.start_ep, self.params.num_epoch+self.start_ep):
            print("Epoch:",epoch)
            # metrics
            if self.params.reg_head:
                U, V = self.compute_projection()
                task_loss = self.train_epoch('reg')
                valid_loss = self.eval_epoch('reg')
            else:
                if self.params.lstm_only:
                    task_loss = self.train_epoch('reg')
                    U, V = self.compute_projection()
                    valid_loss = self.eval_epoch('reg')
                else:
                    train_loss = self.train_epoch('graph')
                    U, V = self.compute_projection()
                    # task_loss = self.train_epoch('task')
                    valid_loss = self.eval_epoch()
                torch.save({'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, self.params.model_file_name)

    def compute_projection(self):
        print('Epoch:project')
        text_embs = []
        graph_embs = []
        metrics, preprocess, dataloader, data_func, forward_func, loss_func, score_func = self.prepare_project_model()
        pbar = tqdm(dataloader)
        preprocess(self.train_set.g)
        with torch.no_grad():
            for batch in pbar:
                batch = batch.ls
                data = move_batch_to_device(batch, self.params.device)
                fdata = data_func(data)
                res = forward_func(self.train_set.g, fdata)
                text_embs.append(res[0])
                graph_embs.append(res[1])
        text_embs = torch.cat(text_embs, dim=0)
        graph_embs = torch.cat(graph_embs, dim=0)
        corr, U, V = self.model.corr_eval.loss(graph_embs, text_embs)
        self.model.U = U
        self.model.V = V
        return U, V

    def prepare_project_model(self):
        metrics = {
            'mse_loss':0
        }
        def data_func(data):
            return (data[1], data[0])
        
        def preprocess(graph):
            self.model.eval()
            with torch.no_grad():
                self.model.graph_update(graph)
        
        def forward_func(g, data):
            gn_repr = self.model.get_graph_embedding(g,data[1])
            ft_repr = self.model.text_update(data[0])
            return ft_repr, gn_repr
        
        dataloader = DataLoader(self.train_set, batch_size=64, num_workers=4, pin_memory=True, shuffle=True, collate_fn=collate_func)
        
        def loss_func(res, data):
            # loss = self.loss(res, data[3].view(-1,1))
            loss = self.cel(res, data[2])
            return loss

        def score_func(loss, res, data, metrics):
            metrics['cross_entropy_loss'] += loss.item()
        
        return metrics, preprocess, dataloader, data_func, forward_func, loss_func, score_func

    def train_epoch(self, type='graph'):
        print('Epoch', type)
        if type == 'graph':
            funcs = self.prepare_graph_model()
        if type == 'task':
            funcs = self.prepare_cotrain_task_model()
        if type == 'reg':
            funcs = self.prepare_reg_model()
        if type == 'reg_graph':
            funcs = self.prepare_reg_graph_model()
        metrics, preprocess, dataloader, data_func, forward_func, loss_func, score_func = funcs
        pbar = tqdm(dataloader)
        preprocess(self.train_set.g)
        for batch in pbar:
            batch = batch.ls
            data = move_batch_to_device(batch, self.params.device)
            fdata = data_func(data)
            res = forward_func(self.train_set.g, fdata)
            loss = loss_func(res, data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                score_func(loss, res, data, metrics)
        formated_res = ""
        for k in metrics:
            formated_res+=k+":"
            formated_res+=str(metrics[k]/len(dataloader))+", "
        print(formated_res)
        return metrics

    def eval_epoch(self, type='graph'):
        print('Eval Epoch', type)
        if type == 'graph':
            funcs = self.prepare_eval_graph_model()
        if type == 'reg':
            funcs = self.prepare_eval_reg_model()
        if type == 'reg_graph':
            funcs = self.prepare_eval_reg_graph_model()
        metrics, preprocess, dataloader, data_func, forward_func, loss_func, score_func = funcs
        # self.reg_model[1].eval()
        pbar = tqdm(dataloader)
        with torch.no_grad():
            v_g = self.valid_set.g
            preprocess(v_g)
            for batch in pbar:
                batch = batch.ls
                data = move_batch_to_device(batch, self.params.device)
                fdata = data_func(data)
                res = forward_func(v_g, fdata)
                loss = loss_func(res,data)
                score_func(loss, res, data, metrics)
        formated_res = ""
        for k in metrics:
            formated_res+=k+":"
            formated_res+=str(metrics[k]/len(dataloader))+", "
        print(formated_res)
        return metrics

    def prepare_graph_model(self):
        metrics = {
            'total_cor':0,
            'total_graph_var':0,
            'total_ft_var':0,
            'total_loss':0
        }
        def data_func(data):
            return (data[1], data[0])

        def preprocess(graph):
            self.model.train()
            self.model.use_precomputed_emb = False
            self.model.batch_project = True
            return None

        forward_func = self.model
        
        dataloader = DataLoader(self.train_set, batch_size=4096, num_workers=4,  shuffle=True, collate_fn=collate_func, pin_memory=True)
        
        def loss_func(res, data):
            loss = res[0]
            return loss

        def score_func(loss, res, data, metrics):
            metrics['total_loss'] += res[0].item()
            metrics['total_cor'] += res[0].item()
            metrics['total_graph_var'] += res[1].item()
            metrics['total_ft_var'] += res[2].item()
        
        return metrics, preprocess, dataloader, data_func, forward_func, loss_func, score_func

    def prepare_task_model(self):
        metrics = {
            'cross_entropy_loss':0
        }
        def data_func(data):
            return (data[1], data[0])
        
        def preprocess(graph):
            self.model.train()
            self.model.batch_project = False
            return None
        
        def forward_func(g, data):
            score, graph_var, ft_var, gn_repr, ft_repr  = self.model(g,data)
            g_pred = self.reg_model[1](ft_repr)
            return g_pred
        
        dataloader = DataLoader(self.train_set, batch_size=64, num_workers=4, pin_memory=True, shuffle=True, collate_fn=collate_func)
        
        def loss_func(res, data):
            # loss = self.loss(res, data[3].view(-1,1))
            loss = self.cel(res, data[2])
            return loss

        def score_func(loss, res, data, metrics):
            metrics['cross_entropy_loss'] += loss.item()
        
        return metrics, preprocess, dataloader, data_func, forward_func, loss_func, score_func
    
    def prepare_cotrain_task_model(self):
        metrics = {
            'mse1':0,
            'mse2':0,
        }
        def data_func(data):
            return (data[1], data[0])
        
        def preprocess(graph):
            self.model.eval()
            for m in self.reg_model:
                m.train()
            self.model.batch_project = False
            self.model.use_precomputed_emb = True
            with torch.no_grad():
                self.model.graph_update(graph)
            return None
        
        def forward_func(g, data):
            score, graph_var, ft_var, gn_repr, ft_repr  = self.model(g,data)
            g_pred = self.reg_model[0](gn_repr)
            f_pred = self.reg_model[1](ft_repr)
            return g_pred, f_pred
        
        dataloader = DataLoader(self.train_set, batch_size=64, num_workers=4, pin_memory=True, shuffle=True, collate_fn=collate_func)
        
        def loss_func(res, data):
            loss1 = self.loss(res[0], data[3].view(-1,1))
            loss2 = self.loss(res[1], data[3].view(-1,1))
            # loss1 = self.cel(res[0], data[2])
            # loss2 = self.cel(res[1], data[2])
            return loss1+loss2

        def score_func(loss, res, data, metrics):
            metrics['mse1'] += self.loss(res[0], data[3].view(-1,1)).item()
            metrics['mse2'] += self.loss(res[1], data[3].view(-1,1)).item()

        
        return metrics, preprocess, dataloader, data_func, forward_func, loss_func, score_func

    def prepare_reg_model(self):
        metrics = {
            'mse1':0,
            'mse2':0,
        }
        def data_func(data):
            return (data[1], data[0])
        
        def preprocess(graph):
            self.model.eval()
            for r in self.reg_model:
                r.train()
            self.model.use_precomputed_emb = True
            self.model.batch_project = False
            with torch.no_grad():
                self.model.graph_update(graph)

        def forward_func(g, data):
            score, graph_var, ft_var, gn_repr, ft_repr  = self.model.fast_forward(g,data)
            g_pred = self.reg_model[0](gn_repr)
            f_pred = self.reg_model[1](ft_repr)
            return g_pred, f_pred
        
        dataloader = DataLoader(self.train_set, batch_size=64, num_workers=4, pin_memory=True, shuffle=True, collate_fn=collate_func)
        
        def loss_func(res, data):
            loss1 = self.loss(res[0], data[3].view(-1,1))
            loss2 = self.loss(res[1], data[3].view(-1,1))
            return loss1+loss2

        def score_func(loss, res, data, metrics):
            metrics['mse1'] += self.loss(res[0], data[3].view(-1,1)).item()
            metrics['mse2'] += self.loss(res[1], data[3].view(-1,1)).item()
        
        return metrics, preprocess, dataloader, data_func, forward_func, loss_func, score_func
    
    def prepare_reg_graph_model(self):
        metrics = {
            'mse_loss':0
        }
        def data_func(data):
            return (data[1], data[0])
        
        def preprocess(graph):
            self.model.eval()
            self.reg_model.train()
            with torch.no_grad():
                self.model.graph_update(graph)

        def forward_func(g, data):
            score, graph_var, ft_var, gn_repr, ft_repr  = self.model.fast_forward(g,data)
            g_pred = self.reg_model(gn_repr)
            return g_pred
        
        dataloader = DataLoader(self.train_set, batch_size=64, num_workers=4, pin_memory=True, shuffle=True, collate_fn=collate_func)
        
        def loss_func(res, data):
            loss = self.loss(res, data[3].view(-1,1))
            return loss

        def score_func(loss, res, data, metrics):
            metrics['mse_loss'] += loss.item()
        
        return metrics, preprocess, dataloader, data_func, forward_func, loss_func, score_func

    def prepare_eval_graph_model(self):
        metrics = {
            'total_graph_var':0,
            'total_ft_var':0,
            'total_loss':0,
            'mse1':0,
            'mse2':0
        }
        def data_func(data):
            return (data[1], data[0])

        def preprocess(graph):
            self.model.eval()
            for r in self.reg_model:
                r.eval()
            self.model.graph_update(graph)
            self.model.batch_project = False
            self.model.use_precomputed_emb = True
        
        def forward_func(g, data):
            score, graph_var, ft_var, gn_repr, ft_repr = self.model.fast_forward(g, data)
            g_pred = self.reg_model[0](gn_repr)
            f_pred = self.reg_model[1](ft_repr)
            return score,g_pred, f_pred, graph_var, ft_var
        
        dataloader = DataLoader(self.valid_set, batch_size=512, num_workers=4,  shuffle=True, collate_fn=collate_func)
        
        def loss_func(res, data):
            loss = res[0]
            rloss = self.loss(res[1], data[3].view(-1,1))
            lloss = self.loss(res[2], data[3].view(-1,1))
            # rloss = self.cel(res[1], data[2])
            return [loss, rloss, lloss]

        def score_func(loss, res, data, metrics):
            metrics['total_loss'] += res[0].item()
            metrics['total_graph_var'] += res[3].item()
            metrics['total_ft_var'] += res[4].item()
            metrics['mse1'] += loss[1].item()
            metrics['mse2'] += loss[2].item()
            
        
        return metrics, preprocess, dataloader, data_func, forward_func, loss_func, score_func

    def prepare_eval_reg_model(self):
        metrics = {
            'mse1':0,
            'mse2':0,
        }
        def data_func(data):
            return (data[1], data[0])
        
        def preprocess(graph):
            self.model.eval()
            for r in self.reg_model:
                r.train()
            self.model.use_precomputed_emb = True
            self.model.batch_project = False
            self.model.graph_update(graph)

        def forward_func(g, data):
            score, graph_var, ft_var, gn_repr, ft_repr  = self.model.fast_forward(g,data)
            g_pred = self.reg_model[0](gn_repr)
            f_pred = self.reg_model[1](ft_repr)
            return g_pred, f_pred
        
        dataloader = DataLoader(self.train_set, batch_size=64, num_workers=4, pin_memory=True, shuffle=True, collate_fn=collate_func)
        
        def loss_func(res, data):
            loss1 = self.loss(res[0], data[3].view(-1,1))
            loss2 = self.loss(res[1], data[3].view(-1,1))
            return loss1+loss2

        def score_func(loss, res, data, metrics):
            metrics['mse1'] += self.loss(res[0], data[3].view(-1,1)).item()
            metrics['mse2'] += self.loss(res[1], data[3].view(-1,1)).item()
        
        return metrics, preprocess, dataloader, data_func, forward_func, loss_func, score_func

    def prepare_eval_reg_graph_model(self):
        metrics = {
            'mse_loss':0
        }
        def data_func(data):
            return (data[1], data[0])
        
        def preprocess(graph):
            self.model.eval()
            self.reg_model.train()
            self.model.graph_update(graph)

        def forward_func(g, data):
            score, graph_var, ft_var, gn_repr, ft_repr  = self.model.fast_forward(g,data)
            g_pred = self.reg_model(gn_repr)
            return g_pred
        
        dataloader = DataLoader(self.valid_set, batch_size=64, num_workers=4, pin_memory=True, shuffle=True, collate_fn=collate_func)
        
        def loss_func(res, data):
            loss = self.loss(res, data[3].view(-1,1))
            return loss

        def score_func(loss, res, data, metrics):
            metrics['mse_loss'] += loss.item()
        
        return metrics, preprocess, dataloader, data_func, forward_func, loss_func, score_func