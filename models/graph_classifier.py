from os import link
from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils.utils import SmartTimer
from models.basic_models import FCLayers
from models.RGCN import RGCN, MultiRGCN, CycleRGCN

class GraphClassifierWhole(nn.Module):
    def __init__(self, params, relation2id):
        super().__init__()

        self.params = params
        self.gnn = RGCN(params) 
        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)

        self.out_emb_dim = self.params.emb_dim*self.params.num_gcn_layers

        self.timer = SmartTimer(False)

        self.num_emb = 2 + self.params.use_mid_repr

        self.geodesic_fc = FCLayers(2, self.out_emb_dim, [64,self.out_emb_dim])

        self.link_fc = FCLayers(3, self.num_emb * self.out_emb_dim + self.params.rel_emb_dim +self.params.use_dist_emb, [256, 256, 1])
        self.node_embedding = None
        self.use_only_embedding = False

        if self.params.use_lstm:
            self.lstm = nn.LSTM(self.out_emb_dim, hidden_size=self.out_emb_dim, batch_first=True)
        self.forward_list = torch.nn.ModuleList()
        self.forward_list.append(self.rel_emb)
        self.forward_list.append(self.geodesic_fc)
        self.forward_list.append(self.link_fc)

    def save_embedding(self, emb):
        self.node_embedding = emb

    def embedding_only_mode(self, state=True):
        self.use_only_embedding = state

    def forward(self, g, data):
        links, dist, geodesic = data
        if self.use_only_embedding:
            repr = g.ndata['repr']
        else:
            repr = self.process_graph(g)
        return self.predict_link(repr, links, dist, geodesic)

    def process_graph(self, g):
        repr = self.gnn(g)
        return repr

    def extract_repr(self, repr, links, geodesic):
        head_repr = repr[links[:,0]]
        tail_repr = repr[links[:,2]]
        rel_repr = self.rel_emb(links[:,1])
        mid_repr = repr[torch.abs(geodesic)]*torch.sign(geodesic+1).unsqueeze(3)
        return head_repr, tail_repr, rel_repr, mid_repr

    def process_geodesic(self, mid_repr):
        if self.params.use_lstm:
            N, M, P, H = mid_repr.size()
            mid_repr, _ = self.lstm(mid_repr.view(N*M, P, H))
            mid_repr = mid_repr[:,0,:].view(N,M,H)
        else:
            mid_repr = mid_repr.sum(dim=-2)
        return mid_repr

    def predict_link(self, repr, links, dist, geodesic):
        self.timer.record()
        head_repr, tail_repr, rel_repr, mid_repr = self.extract_repr(repr, links, geodesic)
        self.timer.cal_and_update('org')
        mid_repr = self.process_geodesic(mid_repr)
        mid_repr = self.geodesic_fc(mid_repr)
        mid_repr = mid_repr.mean(dim=-2)
        # mid_repr = mid_repr.max(dim=-2)[0]
        self.timer.cal_and_update('ct')

        link_emb = [head_repr, tail_repr, rel_repr]
        if self.params.use_mid_repr:
            link_emb += [mid_repr]
        if self.params.use_dist_emb:
            link_emb += [dist.unsqueeze(1)]
        g_rep = torch.cat(link_emb, dim=1)
        output = self.link_fc(g_rep)
        self.timer.cal_and_update('fc')

        return output

class DistSupervisedGC(GraphClassifierWhole):
    def __init__(self, params, relation2id):
        super().__init__(params, relation2id)

        self.out_emb_to_dist = FCLayers(1, self.out_emb_dim, [1])
    
    def predict_link(self, repr, links, dist, geodesic):
        self.timer.record()
        head_repr, tail_repr, rel_repr, mid_repr = self.extract_repr(repr, links, geodesic)
        self.timer.cal_and_update('org')
        mid_repr = self.process_geodesic(mid_repr)
        mid_repr = self.geodesic_fc(mid_repr)
        # mid_repr = mid_repr.mean(dim=-2)
        mid_repr = mid_repr.max(dim=-2)[0]
        self.timer.cal_and_update('ct')
        dist = self.out_emb_to_dist(mid_repr)

        link_emb = [head_repr, tail_repr, rel_repr]
        if self.params.use_mid_repr:
            link_emb += [mid_repr]
        if self.params.use_dist_emb:
            link_emb += [dist.unsqueeze(1)]
        g_rep = torch.cat(link_emb, dim=1)
        output = self.link_fc(g_rep)
        self.timer.cal_and_update('fc')

        return output, dist

class MultiGraphClassifier(GraphClassifierWhole):
    def __init__(self, params, relation2id):
        super(GraphClassifierWhole,self).__init__()
        self.params = params
        if params.use_cycle_edge:
            self.gnn = CycleRGCN(params)
        else:
            self.gnn = MultiRGCN(params)
        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)
        self.out_emb_dim = self.params.emb_dim*self.params.num_gcn_layers
        self.num_emb = 2 + 2*self.params.use_mid_repr
        self.geodesic_fc = FCLayers(2, self.out_emb_dim, [128,self.out_emb_dim])
        self.edge_geodesic_fc = FCLayers(2, self.out_emb_dim, [128,self.out_emb_dim])
        self.link_fc = FCLayers(3, self.num_emb * self.out_emb_dim + self.params.rel_emb_dim +self.params.use_dist_emb, [256, 256, 1])
        self.node_embedding = None
        self.use_only_embedding = False
        self.forward_list = torch.nn.ModuleList()
        self.forward_list.append(self.rel_emb)
        self.forward_list.append(self.geodesic_fc)
        self.forward_list.append(self.edge_geodesic_fc)
        self.forward_list.append(self.link_fc)
        if self.params.use_lstm:
            self.lstm = nn.LSTM(self.out_emb_dim, hidden_size=self.out_emb_dim, batch_first=True)
            self.edge_lstm = nn.LSTM(self.out_emb_dim, hidden_size=self.out_emb_dim, batch_first=True)
            self.forward_list.append(self.lstm)
            self.forward_list.append(self.edge_lstm)
    
    def process_graph(self, g):
        repr, edge_repr = self.gnn(g)
        return repr, edge_repr

    def forward(self, g, data):
        links, dist, geodesic = data
        if self.use_only_embedding:
            repr = (g.ndata['repr'], g.edata['edge_repr'])
        else:
            repr = self.process_graph(g)
        return self.predict_link(repr, links, dist, geodesic)
    
    def predict_link(self, repr, links, dist, geodesic):
        node_repr, edge_repr = repr
        node_geo, edge_geo = geodesic
        head_repr, tail_repr, rel_repr, mid_repr = self.extract_repr(node_repr, links, node_geo)
        mid_edge_repr = edge_repr[torch.abs(edge_geo)]*torch.sign(edge_geo+1).unsqueeze(3)
        mid_repr = self.process_geodesic(mid_repr)
        mid_repr = self.geodesic_fc(mid_repr)
        mid_repr = mid_repr.max(dim=-2)[0]
        mid_edge_repr = mid_edge_repr.sum(dim=-2)
        mid_edge_repr = self.edge_geodesic_fc(mid_edge_repr)
        mid_edge_repr = mid_edge_repr.max(dim=-2)[0]
        link_emb = [head_repr, tail_repr, rel_repr]
        if self.params.use_mid_repr:
            link_emb += [ mid_repr, mid_edge_repr]
        if self.params.use_dist_emb:
            link_emb += [dist.unsqueeze(1)]
        g_rep = torch.cat(link_emb, dim=1)
        output= self.link_fc(g_rep)

        return output