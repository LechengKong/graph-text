from os import link
import torch
import numpy as np
import scipy.sparse as ssp
from torch.utils.data import Dataset
import math
from utils.graph_utils import *

def find_edge(graph, geodesics):
    head = geodesics[:,:,:-1]
    tail = geodesics[:,:,1:]
    # if np.sum(head-424==0)>0:
    #     x,y,z = np.nonzero(head-424==0)
    #     print(link_arr[x])
    #     print('head', head[x,y])
    #     print('tail',tail[x,y])
    a,b,c = head.shape
    x,y,z = np.meshgrid(np.arange(a),np.arange(b),np.arange(c),indexing='ij')
    edges = np.stack([head, tail, x, y, z], axis=-1).reshape(-1,5)
    edges = edges[np.logical_and(edges[:,0]!=-1, edges[:,1]!=-1)]
    e_ids = graph.edge_ids(edges[:,0], edges[:,1])
    organized_e_id = np.zeros(geodesics.shape, dtype=int)-1
    organized_e_id[edges[:,2], edges[:,3], edges[:,4]] = e_ids
    return organized_e_id


class BaseDataSet(Dataset):
    def __init__(self):
        self.data = np.arange(2000).reshape(-1,2)
        self.label = np.arange(1000)/10

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


class NodeTextDataset(Dataset):
    def __init__(self, data, graph):
        self.data = data
        self.num_sample = len(data)
        self.g = graph

    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        node_id, context, ane_type, los = self.data[index]
        return node_id, context, int(ane_type), math.log(los+1)