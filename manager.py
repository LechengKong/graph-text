import torch
from torch_utils import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score
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
        # self.train_set.g.ndata['node_emb'].requires_grad = True
        if train_reg:
            op_param = [p for p in self.reg_model.parameters()]
            self.optimizer = torch.optim.SGD(op_param, lr=self.params.lr)
            self.loss = torch.nn.MSELoss()
            state_d = torch.load(self.params.model_file_name, map_location=self.params.device)
            self.model.load_state_dict(state_d['state_dict'])
            print('model loaded')
        else:
            op_param = [p for p in self.model.parameters()]
            op_param += [self.train_set.g.ndata['node_emb']]
            self.optimizer = torch.optim.SGD(op_param, lr=self.params.lr)
            self.cel = torch.nn.CrossEntropyLoss()
            if self.params.retrain:
                state_d = torch.load(self.params.model_file_name, map_location=self.params.device)
                self.optimizer.load_state_dict(state_d['optimizer'])
                self.model.load_state_dict(state_d['state_dict'])
                self.start_ep = state_d['epoch']+1
    
    def train(self):
        for epoch in range(self.start_ep, self.params.num_epoch+self.start_ep):
            print("Epoch:",epoch)
            if self.params.reg_head:
                task_loss = self.reg_epoch()
                valid_loss = self.reg_epoch_eval()
            else:
                if self.params.lstm_only:
                    task_loss = self.task_epoch()
                    valid_loss = self.forward_epoch()
                else:
                    task_loss = self.task_epoch()
                    train_loss = self.update_epoch()
                    valid_loss = self.forward_epoch()
                torch.save({'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, self.params.model_file_name)

    def reg_epoch(self):
        total_loss = 0
        self.model.eval()
        self.model.requires_grad=False
        self.reg_model.train()
        dataloader = DataLoader(self.train_set, batch_size=64,  num_workers=16,  shuffle=True, collate_fn=collate_func)
        pbar = tqdm(dataloader)
        for batch in pbar:
            g_node, context, context_len, code_desc, ct_len, ane_type, los = move_batch_to_device(batch, self.params.device)
            ft_out, ft_repr = self.model.mlp_update(context, context_len)
            out = self.reg_model(ft_out)
            print(los)
            print(out)
            loss = self.loss(out,los.view(-1,1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                total_loss += loss.item()
        # print(self.train_set.g.ndata['node_emb'])
        print("loss:",total_loss/len(dataloader))
        return total_loss

    def reg_epoch_eval(self):
        total_loss = 0
        self.model.eval()
        self.model.requires_grad=False
        self.reg_model.eval()
        dataloader = DataLoader(self.valid_set, batch_size=64,  num_workers=16, shuffle=True, collate_fn=collate_func)
        pbar = tqdm(dataloader)
        with torch.no_grad():
            for batch in pbar:
                g_node, context, context_len, code_desc, ct_len, ane_type, los = move_batch_to_device(batch, self.params.device)
                ft_out, ft_repr = self.model.mlp_update(context, context_len)
                out = self.reg_model(ft_out)
                loss = self.loss(out,los.view(-1,1))
                total_loss += loss.item()
        # print(self.train_set.g.ndata['node_emb'])
        print("loss:",total_loss/len(dataloader))
        return total_loss

    def update_epoch(self):
        total_cor = 0
        total_graph_var = 0
        total_ft_var = 0
        total_closs = 0
        total_loss = 0
        accuracy = 0
        self.model.train()
        dataloader = DataLoader(self.train_set, batch_size=self.params.batch_size,  num_workers=16,  shuffle=True, collate_fn=collate_func)
        pbar = tqdm(dataloader)
        # with torch.autograd.detect_anomaly():
        for batch in pbar:
            g_node, context, context_len, code_desc, ct_len, ane_type, los = move_batch_to_device(batch, self.params.device)
            score, graph_var, ft_var, ft_repr = self.model((self.train_set.g, context, g_node, context_len))
            # ft_repr = self.model.mlp_update(context, context_len)
            # closs = self.cel(ft_repr, ane_type)
            loss = score
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                total_loss += loss.item()
                total_cor+=score.item()
                # total_closs+=closs.item()
                total_graph_var += graph_var.item()
                total_ft_var += ft_var.item()
                pred = torch.argmax(ft_repr, dim=1).detach().cpu().numpy()
                target = ane_type.detach().cpu().numpy()
                # print(precision_score(target,pred, average=None))
                accuracy+=(torch.sum(torch.argmax(ft_repr, dim=1)==ane_type).item())/len(ane_type)
        # print(self.train_set.g.ndata['node_emb'])
        print("loss:",total_loss/len(dataloader),"corr",total_cor/len(dataloader),"tloss",total_closs/len(dataloader),"acc",accuracy/len(dataloader),"graph var:",total_graph_var/len(dataloader),"ft var:",total_ft_var/len(dataloader))
        return total_loss

    def task_epoch(self):
        total_ft_var = 0
        total_loss = 0
        accuracy = 0
        self.model.train()
        dataloader = DataLoader(self.train_set, batch_size=64, num_workers=16, pin_memory=True, shuffle=True, collate_fn=collate_func)
        pbar = tqdm(dataloader)
        # with torch.autograd.detect_anomaly():
        for batch in pbar:
            g_node, context, context_len, code_desc, ct_len, ane_type, los = move_batch_to_device(batch, self.params.device)
            ft_out, ft_repr = self.model.mlp_update(context, context_len)
            closs = self.cel(ft_repr, ane_type)
            loss = closs
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                total_loss += loss.item()
                pred = torch.argmax(ft_repr, dim=1).detach().cpu().numpy()
                target = ane_type.detach().cpu().numpy()
                # print(torch.argmax(ft_repr, dim=1))
                # print(precision_score(target,pred, average=None))
                accuracy+=(torch.sum(torch.argmax(ft_repr, dim=1)==ane_type).item())/len(ane_type)
        print("loss:",total_loss/len(dataloader),"acc",accuracy/len(dataloader))
        return total_loss

    def forward_epoch(self):
        total_graph_var = 0
        total_ft_var = 0
        total_loss = 0
        accuracy = 0
        self.model.eval()
        dataloader = DataLoader(self.valid_set, batch_size=self.params.batch_size,  num_workers=16, shuffle=True, collate_fn=collate_func)
        pbar = tqdm(dataloader)
        with torch.no_grad():
            v_g = self.valid_set.g
            self.model.graph_update(v_g)
            for batch in pbar:
                g_node, context, context_len, code_desc, ct_len, ane_type,los = move_batch_to_device(batch, self.params.device)
                score, graph_var, ft_var, ft_repr = self.model.fast_forward((v_g, context, g_node,context_len))
                # print(torch.argmax(ft_repr, dim=1))
                loss = score
                total_loss += loss.item()
                total_graph_var += graph_var.item()
                total_ft_var += ft_var.item()
                accuracy+=(torch.sum(torch.argmax(ft_repr, dim=1)==ane_type).item())/len(ane_type)
        print("loss:",total_loss/len(dataloader),"accuracy",accuracy/len(dataloader),"graph var:",total_graph_var/len(dataloader),"ft var:",total_ft_var/len(dataloader))
        return total_loss