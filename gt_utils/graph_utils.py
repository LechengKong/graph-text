import torch
import dgl
import numpy as np

def construct_reverse_graph_from_edges(edges, n_entities, dim=1, num_rels=4):
    g = dgl.graph((np.concatenate((edges[0],edges[2])), np.concatenate((edges[2],edges[0]))), num_nodes=n_entities)
    g.edata['type'] = torch.tensor(np.concatenate([edges[1], edges[1]+num_rels]), dtype=torch.int32)
    g.ndata['feat'] = torch.ones([n_entities, dim], dtype=torch.float32)
    return g