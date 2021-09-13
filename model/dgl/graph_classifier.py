from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifierWhole(nn.Module):
    def __init__(self, params):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params

        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)

    def forward(self, data):
        g, node_id = data
        self.graph_update(g)
        return self.mlp_update(g, node_ids)

    def graph_update(self, g):
        self.gnn(g)

    def mlp_update(self, g, node_ids):
        node_repr = g.ndata['repr'][node_ids]

        return node_repr
