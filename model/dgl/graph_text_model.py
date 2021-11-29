import dgl
import numpy as np
import torch
import scipy
import torch.nn as nn
import time
from .rgcn_model import RGCN

from torch.nn import LSTM


from torch.autograd import Function

def sqrtm(mat):
    m = mat.detach().cpu().numpy().astype(np.float_)
    sqrtmat = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(mat)
    return torch.inverse(sqrtmat)


class CorrelationFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, H1, H2, reg1=0.001, reg2 = 0.001, k=20):
        m,d = H1.size()
        H1hat = H1 - H1.mean(dim=0)
        H2hat = H2 - H2.mean(dim=0)
        I = torch.eye(d, device=H1.device)
        sigma12 = (torch.mm(torch.transpose(H1hat, 0, 1), H2hat))/(m-1)
        sigma11 = (torch.mm(torch.transpose(H1hat, 0, 1), H1hat))/(m-1) + reg1*I
        sigma22 = (torch.mm(torch.transpose(H2hat, 0, 1), H2hat))/(m-1) + reg2*I
        sigma11sqrt = sqrtm(sigma11)
        sigma22sqrt = sqrtm(sigma22)
        T = torch.mm(torch.mm(sigma11sqrt, sigma12), sigma22sqrt)
        u,s,vh = torch.linalg.svd(T)
        # print(s)
        ctx.save_for_backward(H1hat, H2hat, sigma11sqrt, sigma22sqrt, u, s, vh, H1)
        return torch.sum(s[:k])

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        H1hat, H2hat, sigma11sqrt, sigma22sqrt, u, s, vh, H1 = ctx.saved_tensors
        m,d = H1.size()
        grad_H1 = grad_H2 = grad_reg1 = grad_reg2 = grad_k = None

        nabla12 = torch.mm(torch.mm(sigma11sqrt, u), torch.mm(vh,sigma22sqrt))
        nabla21 = torch.mm(torch.mm(sigma22sqrt, u), torch.mm(vh,sigma11sqrt))
        nabla11 = torch.mm(torch.mm(torch.mm(sigma11sqrt, u), torch.mm(torch.diag(s), torch.transpose(u, 0, 1))), sigma11sqrt)
        nabla22 = torch.mm(torch.mm(torch.mm(sigma22sqrt, u), torch.mm(torch.diag(s), torch.transpose(u, 0, 1))), sigma22sqrt)
        if ctx.needs_input_grad[0]:
            grad_H1 = grad_output*(H2hat.mm(torch.transpose(nabla12,0,1))-H1hat.mm(nabla11))/(m-1)
        if ctx.needs_input_grad[1]:
            grad_H2 = grad_output*(H1hat.mm(torch.transpose(nabla21,0,1))-H2hat.mm(nabla22))/(m-1)

        return grad_H1, grad_H2, grad_reg1, grad_reg2, grad_k


class GraphTextModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, ft_rnn_size, ct_rnn_size, params):
        super(GraphTextModel, self).__init__()
        self.params = params
        self.gnn = RGCN(params)
        self.gnn_list = nn.ModuleList()
        self.param_set_dim = self.params.emb_dim*self.params.num_gcn_layers
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        if self.params.cbow_only:
            self.linear1 = nn.Linear(embedding_dim, 128)
        else:
            self.linear1 = nn.Linear(self.param_set_dim, 64)
        self.activation_function1 = nn.ReLU()
        
        self.linear2 = nn.Linear(64, self.params.out_dim)
        self.gnn_list.append(self.gnn)
        self.gnn_list.append(self.linear1)
        self.gnn_list.append(self.linear2)

        self.rnn_list = nn.ModuleList()

        self.ft_rnn = LSTM(self.embedding_dim, self.params.out_dim, batch_first=True)
        # self.h0 = torch.ones((1, self.params.batch_size,  self.params.out_dim), device=self.params.device)
        # self.c0 = torch.ones((1, self.params.batch_size,  self.params.out_dim), device=self.params.device)
        self.corr_eval = CorrelationFunction.apply

        self.ft_linear = nn.Linear(self.params.out_dim, self.params.y_dim)
        self.rnn_list.append(self.ft_rnn)
        self.rnn_list.append(self.ft_linear)
        

    def forward(self, data):
        g, context, node_ids, context_len = data
        self.graph_update(g)
        return self.fast_forward(data)

    def fast_forward(self, data):
        g, context, node_ids, context_len = data
        graph_repr = self.get_graph_embedding(g, node_ids)
        ft_repr, ft_out = self.mlp_update(context, context_len)
        # print(graph_repr)
        # print(torch.var(ft_repr, dim=0).size())
        # print(torch.sqrt(torch.sum((graph_repr-ft_repr)**2, dim=1)))
        return -self.corr_eval(graph_repr, ft_repr, self.params.mat_reg_c[0], self.params.mat_reg_c[1]), torch.var(graph_repr, dim=0).mean(), torch.var(ft_repr, dim=0).mean(), ft_out

    def graph_update(self, g):
        if self.params.use_ct:
            g.ndata['feat'] = torch.cat([self.embeddings(g.ndata['cont_text']).mean(dim=1),
                                            g.ndata['node_emb']], dim=1)
        else:
            g.ndata['feat'] = g.ndata['node_emb']
        self.gnn(g)

    def get_graph_embedding(self, g, node_ids):
        node_repr = g.ndata['repr'][node_ids]
        node_repr = self.linear1(node_repr.view(-1, self.param_set_dim))
        node_repr = self.activation_function1(node_repr)
        node_repr = self.linear2(node_repr)
        # return g.ndata['node_emb'][node_ids]
        return node_repr

    def mlp_update(self, context, context_len):
        embeds = self.embeddings(context)
        ft_repr, (_,_) = self.ft_rnn(embeds)
        ft_out = self.ft_linear(ft_repr[:,0,:])
        return ft_repr[:,0,:], ft_out

    def freeze_rnn(self):
        for layer in self.rnn_list:
            for p in layer.parameters():
                p.requires_grad = False

    def unfreeze_rnn(self):
        for layer in self.rnn_list:
            for p in layer.parameters():
                p.requires_grad = True

    def freeze_gnn(self):
        for layer in self.gnn_list:
            for p in layer.parameters():
                p.requires_grad = False

    def unfreeze_gnn(self):
        for layer in self.gnn_list:
            for p in layer.parameters():
                p.requires_grad = True
    
    def rnn_params(self):
        return [p for p in self.rnn_list.parameters()]

    def gnn_params(self):
        return [p for p in self.gnn_list.parameters()]


class FCLayers(nn.Module):
    def __init__(self, layers, input_dim, h_units, activation=torch.nn.functional.relu):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        for i in range(layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, h_units[i]))
            else:
                self.layers.append(nn.Linear(h_units[i-1], h_units[i]))

    def forward(self, x):
        output = x
        for i, layer in enumerate(self.layers):
            output = layer(output)
            if i < len(self.layers)-1:
                output = self.activation(output)
        return output
