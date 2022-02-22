import imp
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gnn_layers import RGCNLayer, EdgeMessageLayer
from models.basic_models import FCLayers
from aggregators.aggregator import *


class RGCN(nn.Module):
    def __init__(self, params):
        super(RGCN, self).__init__()
        self.params = params
        self.inp_dim = params.inp_dim
        self.emb_dim = params.emb_dim
        self.attn_rel_emb_dim = params.attn_rel_emb_dim
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        self.num_bases = params.num_bases
        self.num_hidden_layers = params.num_gcn_layers
        self.dropout = params.dropout
        self.edge_dropout = params.edge_dropout
        self.agg_type = params.gnn_agg_type
        self.has_attn = params.has_attn
        if params.edge_gnn:
            self.layer_type = EdgeMessageLayer
        else:
            self.layer_type = RGCNLayer

        self.device = params.device

        if self.has_attn:
            self.attn_rel_emb = nn.Embedding(self.aug_num_rels, self.attn_rel_emb_dim, sparse=False)
        else:
            self.attn_rel_emb = None

        # initialize aggregators for input and hidden layers
        if params.gnn_agg_type == "sum":
            self.aggregator = SumAggregator(self.emb_dim)
            # self.aggregator = NormalizedSumAggregator(self.emb_dim)
        elif params.gnn_agg_type == "mlp":
            self.aggregator = MLPAggregator(self.emb_dim)
        elif params.gnn_agg_type == "gru":
            self.aggregator = GRUAggregator(self.emb_dim)
        elif params.gnn_agg_type == "pna":
            self.aggregator = PNAAggregator(self.emb_dim)
        elif params.gnn_agg_type == "edge":
            self.aggregator = EdgeReprAggregator(self.emb_dim)

        # initialize basis weights for input and hidden layers
        # self.input_basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.emb_dim))
        # self.basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.emb_dim, self.emb_dim))

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers - 1):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)

    def build_input_layer(self):
        return self.layer_type(self.inp_dim,
                         self.emb_dim,
                         self.aggregator,
                         self.attn_rel_emb_dim,
                         self.aug_num_rels,
                         self.num_bases,
                         activation=F.relu,
                         dropout=self.dropout,
                         edge_dropout=self.edge_dropout,
                         is_input_layer=True,
                         has_attn=self.has_attn)

    def build_hidden_layer(self):
        return self.layer_type(self.emb_dim,
                         self.emb_dim,
                         self.aggregator,
                         self.attn_rel_emb_dim,
                         self.aug_num_rels,
                         self.num_bases,
                         activation=F.relu,
                         dropout=self.dropout,
                         edge_dropout=self.edge_dropout,
                         has_attn=self.has_attn)

    def forward(self, g):
        h = g.ndata['feat']
        mask = g.edata['mask']
        repr = self.message(g, h, mask)
        return repr

    def message(self, g, h, mask=None):
        repr = None
        for layer in self.layers:
            h = layer(g, h, mask=mask)
            if layer.is_input_layer:
                repr = h
            else:
                repr = torch.cat([repr,h], dim = 1)
        return repr

class MultiRGCN(RGCN):
    def __init__(self, params):
        super().__init__(params)
        self.out_emb_dim = self.params.emb_dim*self.params.num_gcn_layers
        self.node2edge_fc = FCLayers(2, self.out_emb_dim*4, [self.out_emb_dim*2, self.out_emb_dim])


    def forward(self, g):
        h = g.ndata['feat']
        masks = None
        reprs = None
        for i in range(self.params.edge_rep):
            p = torch.zeros(int(g.num_edges()/2), 1, device=g.device)+self.params.edge_pick_ratio
            mask = torch.bernoulli(p)
            mask = torch.cat([mask, mask], dim=0)
            mask = g.edata['mask']*mask
            repr = self.message(g, h, mask)
            if masks is None:
                masks = mask
                reprs = repr
            else:
                masks = torch.cat([masks,mask],dim=1)
                reprs = torch.cat([reprs,repr],dim=1)
        with g.local_scope():
            g.edata['masks'] = masks.view(g.num_edges(), self.params.edge_rep, 1)
            # print(g.edata['masks'][:5])
            g.ndata['reprs'] = reprs.view(g.num_nodes(), self.params.edge_rep, -1)
            # print(torch.norm(g.ndata['reprs'][:10], dim=-1))
            def get_edge_repr(edges):
                edges_repr = torch.cat([edges.src['reprs'], edges.dst['reprs']],dim=-1)
                pos_repr = torch.sum(edges_repr*edges.data['masks'], dim=-2)/torch.clamp(edges.data['masks'].sum(dim=-2), min=1)
                neg_repr = torch.sum(edges_repr*torch.logical_not(edges.data['masks']), dim=-2)/torch.clamp(torch.logical_not(edges.data['masks']).sum(dim=-2), min=1)
                full_repr = torch.cat([pos_repr,neg_repr], dim=-1)
                edges_repr = self.node2edge_fc(full_repr)
                return {'edge_repr':edges_repr}
            g.apply_edges(get_edge_repr)
            repr = torch.mean(g.ndata['reprs'], dim=-2)
            edge_repr = g.edata['edge_repr']
        return repr, edge_repr


class CycleRGCN(RGCN):
    def __init__(self, params):
        super().__init__(params)
        self.out_emb_dim = self.params.emb_dim*self.params.num_gcn_layers
        self.node2edge_fc = FCLayers(2, 40, [40, self.out_emb_dim])


    def forward(self, g):
        h = g.ndata['feat']
        masks = []
        cycle_counts = []
        for i in range(self.params.edge_rep):
            p = torch.zeros(int(g.num_edges()/2), 1, device=g.device)+self.params.edge_pick_ratio
            mask = torch.bernoulli(p)
            e_id = torch.nonzero(torch.logical_not(mask), as_tuple=True)[0]
            src, dst = g.find_edges(e_id)
            mask = torch.cat([mask, mask], dim=0)
            mask = g.edata['mask']*mask
            adj_mat = g.adj(scipy_fmt='csr')
            adj_mat = adj_mat.tolil()
            src = src.detach().cpu().numpy().astype(int)
            dst = dst.detach().cpu().numpy().astype(int)
            adj_mat[src, dst] = 0
            adj_mat[dst, src] = 0
            adj_mat = adj_mat.tocsr()
            temp = adj_mat
            cycles = []
            for j in range(10):
                cycles.append(temp.diagonal())
                temp = temp @ adj_mat
            cycles = np.stack(cycles, axis=0)
            cycle_counts.append(cycles.T)
            masks.append(mask)
        cycle_counts = torch.tensor(np.stack(cycle_counts, axis=-2),device=g.device, dtype=torch.float)
        masks = torch.stack(masks,dim=1)
        with g.local_scope():
            g.edata['masks'] = masks
            # print(g.edata['masks'][:5])
            g.ndata['cycle_c'] = F.normalize(torch.log(torch.clamp(cycle_counts, min=1)), dim=0)
            # g.ndata['cycle_c'] = torch.log(torch.clamp(cycle_counts, min=1))
            def get_edge_repr(edges):
                edges_repr = torch.cat([edges.src['cycle_c'], edges.dst['cycle_c']],dim=-1)
                pos_repr = torch.sum(edges_repr*edges.data['masks'], dim=-2)/torch.clamp(edges.data['masks'].sum(dim=-2), min=1)
                neg_repr = torch.sum(edges_repr*torch.logical_not(edges.data['masks']), dim=-2)/torch.clamp(torch.logical_not(edges.data['masks']).sum(dim=-2), min=1)
                full_repr = torch.cat([pos_repr,neg_repr], dim=-1)
                edges_repr = self.node2edge_fc(full_repr)
                return {'edge_repr':edges_repr}
            g.apply_edges(get_edge_repr)
            edge_repr = g.edata['edge_repr']
            mask = g.edata['mask']
            repr = self.message(g, h, mask)
        return repr, edge_repr