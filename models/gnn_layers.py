import torch
import torch.nn as nn
import torch.nn.functional as F

class RGCNLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, aggregator, attn_rel_emb_dim, num_rels, num_bases=-1, bias=None, activation=None, dropout=0.0, edge_dropout=0.0, is_input_layer=False, has_attn=False):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.aggregator = aggregator
        self.activation = activation
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.attn_rel_emb_dim = attn_rel_emb_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        self.has_attn = has_attn

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        self.dropout = nn.Dropout(dropout)

        self.edge_dropout = nn.Dropout(edge_dropout)

        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.out_dim))
        self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        self.self_loop_weight = nn.Parameter(torch.Tensor(self.inp_dim, self.out_dim))

        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    def forward(self, g, h, attn_rel_emb=None, mask=None):
        
        weight = self.weight.view(self.num_bases, self.inp_dim * self.out_dim)
        weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.inp_dim, self.out_dim)
        scale = torch.log(g.in_degrees()+1)
        scale = scale/scale.mean()
        
        def msg_func(edges):
            w = weight.index_select(0, edges.data['type'])
            msg = edges.data['w'] * torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze(1)
            curr_emb = torch.mm(edges.dst['h'], self.self_loop_weight)  # (B, F)

            if self.has_attn:
                e = torch.cat([edges.src['h'], edges.dst['h'], attn_rel_emb(edges.data['type'])], dim=1)
                a = torch.sigmoid(self.B(F.relu(self.A(e))))
            else:
                a = torch.ones((len(edges), 1)).to(device=w.device)

            return {'curr_emb': curr_emb, 'msg': msg, 'alpha': a, 'e_type':edges.data['type'], 'e_count':edges.dst['nei_edge_count'], 'scale':edges.dst['scale']}
        
        with g.local_scope():
            g.ndata['h']=h
            g.ndata['scale']=scale
            if mask is None:
                g.edata['w']=self.edge_dropout(torch.ones(g.number_of_edges(), 1).to(weight.device))
            else:
                g.edata['w']=self.edge_dropout(torch.ones(g.number_of_edges(), 1).to(weight.device))*mask
            g.update_all(msg_func, self.aggregator)

            node_repr = g.ndata['out']
            if self.bias:
                node_repr = node_repr + self.bias
            if self.activation:
                node_repr = self.activation(node_repr)
            if self.dropout:
                node_repr = self.dropout(node_repr)

            return node_repr

class EdgeMessageLayer(RGCNLayer):
    def forward(self, g, h, attn_rel_emb=None, mask=None):
        
        weight = self.weight.view(self.num_bases, self.inp_dim * self.out_dim)
        weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.inp_dim, self.out_dim)
        scale = torch.log(g.in_degrees()+1)
        scale = scale/scale.mean()
        
        def msg_func(edges):
            w = weight.index_select(0, edges.data['type'])
            msg = edges.data['w'] * torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze(1)
            curr_emb = torch.mm(edges.dst['h'], self.self_loop_weight)  # (B, F)

            if self.has_attn:
                e = torch.cat([edges.src['h'], edges.dst['h'], attn_rel_emb(edges.data['type'])], dim=1)
                a = torch.sigmoid(self.B(F.relu(self.A(e))))
            else:
                a = torch.ones((len(edges), 1)).to(device=w.device)
            e_msg = edges.data['w'] * edges.data['edge_repr']

            return {'curr_emb': curr_emb, 'msg': msg, 'alpha': a, 'e_type':edges.data['type'], 'e_count':edges.dst['nei_edge_count'], 'scale':edges.dst['scale'], 'e_repr':e_msg}
        
        with g.local_scope():
            g.ndata['h']=h
            g.ndata['scale']=scale
            if mask is None:
                g.edata['w']=self.edge_dropout(torch.ones(g.number_of_edges(), 1).to(weight.device))
            else:
                g.edata['w']=self.edge_dropout(torch.ones(g.number_of_edges(), 1).to(weight.device))*mask
            g.update_all(msg_func, self.aggregator)

            node_repr = g.ndata['out']
            if self.bias:
                node_repr = node_repr + self.bias
            if self.activation:
                node_repr = self.activation(node_repr)
            if self.dropout:
                node_repr = self.dropout(node_repr)

            return node_repr