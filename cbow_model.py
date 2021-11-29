import torch
import torch.nn as nn
import pickle as pkl
import numpy as np
import dgl
import time
import os.path as osp
from model.dgl.graph_text_model import GraphTextModel, FCLayers
from utils import *
from graph_utils import construct_reverse_graph_from_edges
from manager import Manager
from dataset import NodeTextDataset


EMDEDDING_DIM = 100

class Mem:

    def __init__(self):
        self.data_path = "/project/tantra/jerry.kong/med_data"
        self.model_path = "/project/tantra/jerry.kong/med_model"
        self.num_rels = 4
        self.aug_num_rels = 8
        self.rel_emb_dim = 64
        self.emb_dim = 64
        self.attn_rel_emb_dim = 64
        self.node_emb_dim = 32
        self.add_ht_emb = True
        self.has_attn = False
        self.num_gcn_layers = 2
        self.num_bases = 4
        self.dropout = 0
        self.edge_dropout = 0
        self.gnn_agg_type = 'sum'
        self.inp_dim = 100
        self.cbow_only = False
        self.lstm_only = True
        self.reg_head = True
        self.model_name ='emb_cont_ff' #'lstm_only'   #lg md 'emb_cont_ff'
        self.retrain = False
        self.out_dim = 64
        self.num_epoch = 20
        self.use_ct = False
        self.batch_size = 512
        self.lr = 0.001
        self.mat_reg_c = [0.001,0.001]
        self.y_dim = 2


params = Mem()
if params.cbow_only:
    params.model_file_name = osp.join(params.model_path, params.model_name+"_only.pth")
else:
    params.model_file_name = osp.join(params.model_path,params.model_name+".pth")


if torch.cuda.is_available():
    params.device = torch.device('cpu')
else:
    params.device = torch.device('cpu')

params.inp_dim = params.node_emb_dim+params.use_ct*EMDEDDING_DIM

pc_icd_9 = open_and_load_pickle(osp.join(params.data_path, 'ICD-9-DATA.pkl'))
[w2i, i2w] = open_and_load_pickle(osp.join(params.data_path, 'word-index-map.pkl'))
[e2n,n2e] = open_and_load_pickle(osp.join(params.data_path,'e2n.pkl'))
links = open_and_load_pickle(osp.join(params.data_path,'graph.pkl'))
c2d = open_and_load_pickle(osp.join(params.data_path,'c2d.pkl'))
print("Read data")

vocab_size, padding_index = process_vocab(w2i, i2w)

num_nodes = np.max(np.array(list(e2n.values())))+1

print("Prepare vocab")

max_ft_len = get_max_text_len(pc_icd_9, 1)

print("max text len:", max_ft_len)

max_prc_len = get_max_text_len(pc_icd_9, 0)

print("max procedure code length")

max_ct_len = get_max_text_len(c2d.items(), 1)

print("icd9pc max len:", max_ct_len)

code_text = get_indexed_data(pc_icd_9, max_prc_len, max_ft_len, max_ct_len, e2n, w2i, padding_index)

code_mat = np.array(code_text, dtype=object)
params.y_dim = np.max(code_mat[:,5])+1

train, valid = data_split(code_mat, 0.2)

links = np.array(links)
g = construct_reverse_graph_from_edges(links.T, num_nodes, EMDEDDING_DIM)

emb_pick = prepare_ct_graph(c2d, g.num_nodes(), max_ct_len, padding_index, e2n, w2i)

g.ndata['cont_text'] = torch.LongTensor(emb_pick)
g = g.to(params.device)
g.ndata['node_emb'] = torch.rand((g.num_nodes(), params.node_emb_dim), device = params.device, requires_grad=True)
train_set = NodeTextDataset(train, g)
valid_set = NodeTextDataset(valid, g)
model = GraphTextModel(vocab_size, EMDEDDING_DIM, max_ft_len, max_ct_len,params).to(params.device)
reg_model = FCLayers(3, params.out_dim, [32,16,1])
manager = Manager(train_set, valid_set, model, params, reg_model)
manager.setup(True)
manager.train()
