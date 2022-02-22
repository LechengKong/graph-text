import torch
import torch.nn as nn
import torch.nn.functional as F

class Aggregator(nn.Module):
    def __init__(self, emb_dim):
        super(Aggregator, self).__init__()
        self.emb_dim = emb_dim

    def forward(self, node):
        curr_emb = node.mailbox['curr_emb'][:, 0, :]  # (B, F)
        nei_msg = self.process_message(node)

        new_emb = self.update_embedding(curr_emb, nei_msg)

        return {'out': new_emb}

    def process_message(self, node):
        nei_msg = torch.bmm(node.mailbox['alpha'].transpose(1, 2), node.mailbox['msg']).squeeze(1)  # (B, F)
        return nei_msg

    def update_embedding(self, curr_emb, nei_msg):
        raise NotImplementedError

class PNAAggregator(Aggregator):
    def __init__(self, emb_dim):
        super().__init__(emb_dim)
        self.linear = nn.Linear(13 * emb_dim, emb_dim)

    def process_message(self, node):
        N,M,D = node.mailbox['msg'].size()
        mean = node.mailbox['msg'].mean(dim=-2)
        sq_mean = (node.mailbox['msg']**2).mean(dim=-2)
        max = node.mailbox['msg'].max(dim=-2)[0]
        min = node.mailbox['msg'].min(dim=-2)[0]
        std = (sq_mean-mean**2).clamp(min=1e-8).sqrt()
        features = torch.cat([mean,max,min,std],dim=-1)
        scale = node.mailbox['scale'][0,0:1]
        scale = torch.cat([torch.tensor([1], device=mean.device), scale, 1/scale.clamp(min=1e-3)])
        msg = (features.unsqueeze(-2)*scale.view(1,3,1)).flatten(-2)

        return msg

    def update_embedding(self, curr_emb, nei_msg):
        inp = torch.cat((nei_msg, curr_emb), 1)
        new_emb = F.relu(self.linear(inp))
        return new_emb
        
class EdgeReprAggregator(Aggregator):
    def __init__(self, emb_dim):
        super().__init__(emb_dim)
        self.linear1 = nn.Linear(emb_dim*4, emb_dim)
        self.linear2 = nn.Linear(emb_dim*2, emb_dim)
        self.dropout = nn.Dropout(0.3)
    
    def process_message(self, node):
        msg = torch.cat([node.mailbox['msg'], node.mailbox['e_repr']], dim=-1)
        msg = self.dropout(msg)
        msg = self.linear1(msg)
        msg = F.relu(msg)
        nei_msg = torch.bmm(node.mailbox['alpha'].transpose(1, 2), msg).squeeze(1)
        return nei_msg
    
    def update_embedding(self, curr_emb, nei_msg):
        emb = torch.cat([curr_emb, nei_msg], dim=-1)
        emb = self.linear2(emb)
        emb = F.relu(emb)
        return emb

class NormalizeAggregator(Aggregator):
    def __init__(self, emb_dim):
        super().__init__(emb_dim)
        self.nei_msg_fc = nn.Linear( emb_dim, int(emb_dim/2))
        self.nei_msg_total_fc = nn.Linear( emb_dim, int(emb_dim/2))
    
    def process_message(self, node):
        e_type = node.mailbox['e_type']
        e_count = node.mailbox['e_count'][:, 0, :]
        e_total_count = e_count.sum(dim=-1)
        e_ratio = (1/torch.gather(e_count, -1, e_type)).unsqueeze(1)
        nei_msg = torch.bmm(e_ratio, node.mailbox['msg']).squeeze(1)
        norm_nei_msg = node.mailbox['msg'].sum(dim=-2)/e_total_count.unsqueeze(1)
        nei_msg = torch.cat([self.nei_msg_fc(nei_msg),self.nei_msg_total_fc(norm_nei_msg)],dim=-1)
        return nei_msg

class SumAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(SumAggregator, self).__init__(emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = nei_msg + curr_emb

        return new_emb


class MLPAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(MLPAggregator, self).__init__(emb_dim)
        self.linear = nn.Linear(2 * emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        inp = torch.cat((nei_msg, curr_emb), 1)
        new_emb = F.relu(self.linear(inp))

        return new_emb


class GRUAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(GRUAggregator, self).__init__(emb_dim)
        self.gru = nn.GRUCell(emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = self.gru(nei_msg, curr_emb)

        return new_emb

class NormalizedSumAggregator(NormalizeAggregator, SumAggregator):
    def __init__(self, emb_dim):
        super().__init__(emb_dim)

class SumEdgeAggregator(EdgeReprAggregator, SumAggregator):
    def __init__(self, emb_dim):
        super().__init__(emb_dim)