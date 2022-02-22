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

class cca_loss():
    def __init__(self, outdim_size=20):
        self.outdim_size = outdim_size
        # print(device)

    def loss(self, H1, H2):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        """

        r1 = 1e-4
        r2 = 1e-4
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()
        # assert torch.isnan(H1).sum().item() == 0
        # assert torch.isnan(H2).sum().item() == 0

        o1 = o2 = H1.size(0)

        m = H1.size(1)
#         print(H1.size())

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        # assert torch.isnan(H1bar).sum().item() == 0
        # assert torch.isnan(H2bar).sum().item() == 0

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=H1.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=H1.device)
        # print(SigmaHat11.sum())
        # print(SigmaHat22.sum())
        # print(SigmaHat12.sum())
        assert torch.isnan(SigmaHat11).sum().item() == 0
        assert torch.isnan(SigmaHat12).sum().item() == 0
        assert torch.isnan(SigmaHat22).sum().item() == 0

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.linalg.eigh(SigmaHat11)
        [D2, V2] = torch.linalg.eigh(SigmaHat22)
        # v,c = torch.unique(D1, return_counts=True)
        # print(v[c>1])
        # v,c = torch.unique(D2, return_counts=True)
        # print(v[c>1])
        assert torch.isnan(D1).sum().item() == 0
        assert torch.isnan(D2).sum().item() == 0
        assert torch.isnan(V1).sum().item() == 0
        assert torch.isnan(V2).sum().item() == 0

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        # print(posInd1.size())
        # print(posInd2.size())

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)                                
#         print(Tval.size())

        # just the top self.outdim_size singular values are used
        trace_TT = torch.matmul(Tval.t(), Tval)
        trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(H1.device)) # regularization for more stability
        U, V = torch.linalg.eigh(trace_TT)
        U = torch.where(U>eps, U, (torch.ones(U.shape)*eps).to(H1.device))
        U = U.topk(self.outdim_size)[0]
        corr = torch.sum(torch.sqrt(U))
        U, S, V = torch.svd(Tval)
        U = U[:,:self.outdim_size]
        V = V[:,:self.outdim_size]
        U = torch.matmul(SigmaHat11RootInv, U)
        V = torch.matmul(SigmaHat22RootInv, V)
        return corr, U, V


class GraphTextModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, ft_rnn_size, ct_rnn_size, params):
        super(GraphTextModel, self).__init__()
        self.params = params
        self.gnn = RGCN(params)
        self.gnn_list = nn.ModuleList()
        self.param_set_dim = self.params.emb_dim*self.params.num_gcn_layers
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(self.param_set_dim, 64)
        self.activation_function1 = nn.ReLU()
        
        self.linear2 = nn.Linear(64, self.params.out_dim)
        self.gnn_fc = FCLayers(2, self.param_set_dim, [128, self.params.out_dim])
        self.gnn_list.append(self.gnn)
        self.gnn_list.append(self.gnn_fc)

        self.rnn_list = nn.ModuleList()

        self.ft_rnn = LSTM(self.embedding_dim, self.params.lstm_dim, batch_first=True)
        self.ct_rnn = LSTM(self.embedding_dim, self.embedding_dim, batch_first=True)

        self.rnn_fc = FCLayers(2, self.params.lstm_dim, [64, self.params.out_dim])
        
        # self.corr_eval = CorrelationFunction.apply
        self.corr_eval = cca_loss(self.params.corr_dim)

        self.rnn_list.append(self.ft_rnn)
        self.rnn_list.append(self.rnn_fc)

        self.node_emb = nn.Parameter(torch.Tensor(self.params.num_nodes, self.params.inp_dim))
        nn.init.xavier_uniform_(self.node_emb, gain=nn.init.calculate_gain('relu'))
        self.batch_project = True
        self.use_precomputed_emb = False
        self.U = None
        self.V = None

        

    def forward(self, g, data):
        if not self.use_precomputed_emb:
            self.graph_update(g)
        return self.fast_forward(g, data)

    def fast_forward(self, g, data):
        context, node_ids = data
        graph_repr = self.get_graph_embedding(g, node_ids)
        
        ft_repr = self.text_update(context)

        corr, Ub, Vb = self.corr_eval.loss(graph_repr, ft_repr)
        closs = -corr
        if self.batch_project:
            U = Ub
            V = Vb
        else:
            U = self.U
            V = self.V
        # if self.batch_project:
        #     self.U = Ub
        #     self.V = Vb
        # U = self.U
        # V = self.V
        graph_repr = torch.matmul(graph_repr, U)
        ft_repr = torch.matmul(ft_repr, V)
        # g_var = torch.sqrt(torch.var(graph_repr, dim=0))
        # f_var = torch.sqrt(torch.var(ft_repr, dim=0))
        # g_mean = torch.mean(graph_repr, dim=0)
        # f_mean = torch.mean(ft_repr, dim=0)
        # cov = torch.sum((graph_repr-g_mean)*(ft_repr-f_mean), dim=0)/(len(graph_repr)-1)
        # bcorr = cov/(g_var*f_var)
        # print(bcorr)
        return closs, torch.var(graph_repr, dim=0).mean(), torch.var(ft_repr, dim=0).mean(), graph_repr, ft_repr

    def graph_update(self, g):
        if self.params.use_ct:
            ct_repr, (_,_) = self.ct_rnn(self.embeddings(g.ndata['cont_text']))
            g.ndata['feat'] = ct_repr[:,0,:]
        else:
            g.ndata['feat'] = self.node_emb
        self.gnn(g)

    def get_graph_embedding(self, g, node_ids):
        node_repr = g.ndata['repr'][node_ids]
        node_repr = torch.flatten(node_repr, start_dim=2)
        m_nodes = torch.sign(node_ids+1)
        node_repr = (node_repr*m_nodes.unsqueeze(2)).sum(dim=1)/m_nodes.sum(dim=1).unsqueeze(1)
        node_repr = self.gnn_fc(node_repr)
        # return g.ndata['node_emb'][node_ids]
        return node_repr

    def text_update(self, context):
        embeds = self.embeddings(context)
        repr, (_,_) = self.ft_rnn(embeds)
        repr = repr[:,0,:]
        # mask = context!=self.params.padding_index
        # repr = torch.sum(embeds*mask.unsqueeze(2), dim=1)
        return self.rnn_fc(repr)
    
    def mlp_update(self, context, g, node_ids):
        # return self.text_update(context)
        return self.get_graph_embedding(g, node_ids)

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
