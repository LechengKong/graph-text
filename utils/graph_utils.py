import numpy as np
import scipy.sparse as ssp
import torch
import dgl
import random
import time

def sample_neg_link(adj, pos_link, num_nodes, sample_size=1, rel_count = 0):
    head, rel, tail = pos_link
    head_arr = np.ones(num_nodes)
    tail_arr = np.ones(num_nodes)
    cur_adj = adj[rel]
    head_neighbor = cur_adj.col[cur_adj.row==head]
    tail_neighbor = cur_adj.row[cur_adj.col==tail]
    tail_arr[head_neighbor] = 0 
    head_arr[tail_neighbor] = 0
    tail_arr[tail] = 0
    tail_arr[head] = 0
    head_arr[tail] = 0
    head_arr[head] = 0
    tail_cans = np.nonzero(tail_arr)[0]
    head_cans = np.nonzero(head_arr)[0]
    tail_sample_num = int(len(tail_cans)/(len(tail_cans)+len(head_cans))*sample_size)
    head_sample_num = sample_size - tail_sample_num
    tail_sample = np.random.choice(tail_cans, tail_sample_num, replace = True)
    head_sample = np.random.choice(head_cans, head_sample_num, replace = True)
    link_arr = np.zeros((sample_size+1, 3), dtype=int)
    link_arr[0] = pos_link
    link_arr[1:,1] = rel
    link_arr[1:tail_sample_num+1, 0] = head
    link_arr[1:tail_sample_num+1, 2] = tail_sample
    link_arr[tail_sample_num+1:, 2] = tail
    link_arr[tail_sample_num+1:, 0] = head_sample
    return link_arr

def sample_filtered_neg_tail(adj, pos_link, num_nodes, sample_size=1, rel_count = 0):
    head, rel, tail = pos_link
    arr = np.arange(num_nodes)
    cur_adj = adj[rel]
    head_neighbor = cur_adj.col[cur_adj.row==head]
    arr[head_neighbor] = 0
    arr[head] = 0
    arr[tail] = 0
    tail_can = np.nonzero(arr)[0]
    link_arr = np.zeros((len(tail_can)+1, 3), dtype=int)
    link_arr[0] = pos_link
    link_arr[1:, 1] = rel
    link_arr[1:, 0] = head
    link_arr[1:, 2] = tail_can
    return link_arr

def sample_filtered_neg_head(adj, pos_link, num_nodes, sample_size=1, rel_count = 0):
    head, rel, tail = pos_link
    arr = np.arange(num_nodes)
    cur_adj = adj[rel]
    tail_neighbor = cur_adj.col[cur_adj.row==tail]
    arr[tail_neighbor] = 0
    arr[head] = 0
    arr[tail] = 0
    head_can = np.nonzero(arr)[0]
    link_arr = np.zeros((len(head_can)+1, 3), dtype=int)
    link_arr[0] = pos_link
    link_arr[1:, 1] = rel
    link_arr[1:, 0] = head_can
    link_arr[1:, 2] = tail
    return link_arr

def construct_reverse_graph_from_edges(edges, n_entities, num_rel):
    g = dgl.graph((np.concatenate((edges[0],edges[2])), np.concatenate((edges[2],edges[0]))), num_nodes=n_entities)
    g.edata['type'] = torch.tensor(np.concatenate((edges[1],edges[1]+num_rel)), dtype=torch.long)
    g.edata['mask'] = torch.tensor(np.ones((len(edges[0])*2,1)), dtype=torch.int32)
    g.edata['oh_type'] = torch.nn.functional.one_hot(g.edata['type'], num_classes=num_rel*2)
    g.update_all(lambda edges: {'msg':edges.data['oh_type'].to(torch.float)}, dgl.function.sum('msg','nei_edge_count'))
    return g

def construct_reverse_homo_graph_from_edges(edges, n_entities):
    g = dgl.graph((np.concatenate((edges[0],edges[2])), np.concatenate((edges[2],edges[0]))), num_nodes=n_entities)
    g.edata['type'] = torch.tensor(np.concatenate((edges[1],edges[1])), dtype=torch.long)
    g.edata['mask'] = torch.tensor(np.ones((len(edges[0])*2,1)), dtype=torch.int32)
    g.update_all(lambda edges: {'msg':torch.ones((len(edges),1), dtype=torch.float)}, dgl.function.sum('msg','nei_edge_count'))
    return g
