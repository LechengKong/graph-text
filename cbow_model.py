import torch
import torch.nn as nn
from torch.optim import Adam
import pickle as pkl
import numpy as np
import dgl
import time
import os.path as osp
import argparse
import random
from model.dgl.graph_text_model import GraphTextModel, FCLayers
from gt_utils.utils import *
from utils.graph_utils import construct_reverse_graph_from_edges
from datasets.dataset import NodeTextDataset
from managers.learner import *
from managers.trainer import *
from managers.manager import Manager


EMDEDDING_DIM = 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='graph-text')

    parser.add_argument("--data_path", type=str, default="/project/tantra/jerry.kong/med_data")
    parser.add_argument("--model_path", type=str, default="/project/tantra/jerry.kong/med_model")
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--num_rels", type=int, default=4)
    parser.add_argument("--num_bases", type=int, default=4)
    parser.add_argument("--attn_rel_emb_dim", type=int, default=64)
    parser.add_argument("--rel_emb_dim", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--num_gcn_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--edge_dropout", type=float, default=0)
    parser.add_argument("--has_attn", type=bool, default=False)
    parser.add_argument("--gnn_agg_type", type=str, default='sum')
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--corr_dim", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--retrain", type=bool, default=False)
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--reg_head", type=bool, default=False)
    parser.add_argument("--lstm_only", type=bool, default=False)
    parser.add_argument("--use_ct", type=bool, default=False)
    parser.add_argument("--node_emb_dim", type=int, default=32)
    parser.add_argument("--lstm_dim", type=int, default=64)
    parser.add_argument("--dry_run", type=bool, default=False)
    parser.add_argument("--num_workers", type=bool, default=4)
    parser.add_argument("--fixed_emb", type=bool, default=False)
    parser.add_argument("--save_model", type=bool, default=True)


    torch.manual_seed(8)
    random.seed(8)
    np.random.seed(8)

    params = parser.parse_args()

    if torch.cuda.is_available():
        params.device = torch.device('cpu')
    else:
        params.device = torch.device('cpu')

    params.inp_dim = params.node_emb_dim
    params.aug_num_rels = params.num_rels*2
    params.model_file_name = "gt.pth"

    if params.fixed_emb:
        params.save_model = False

    pc_icd_9 = open_and_load_pickle(osp.join(params.data_path, 'ICD-9-DATA.pkl'))
    [w2i, i2w] = open_and_load_pickle(osp.join(params.data_path, 'word-index-map.pkl'))
    [e2n,n2e] = open_and_load_pickle(osp.join(params.data_path,'e2n.pkl'))
    links = open_and_load_pickle(osp.join(params.data_path,'graph.pkl'))
    c2d = open_and_load_pickle(osp.join(params.data_path,'c2d.pkl'))

    vocab_size, padding_index = process_vocab(w2i, i2w)
    params.padding_index = padding_index

    num_nodes = np.max(np.array(list(e2n.values())))+1

    max_ft_len = get_max_text_len(pc_icd_9, 1)

    max_prc_len = get_max_text_len(pc_icd_9, 0)

    max_ct_len = get_max_text_len(c2d.items(), 1)
    
    print("max text len:", max_ft_len)
    print("max procedure code length", max_prc_len)
    print("icd9pc max len:", max_ct_len)

    code_text = get_indexed_data(pc_icd_9, max_prc_len, max_ft_len, max_ct_len, e2n, w2i, padding_index)

    code_mat = np.array(code_text, dtype=object)
    params.y_dim = np.max(code_mat[:,2])+1
    params.ft_pred_dim = params.y_dim
    params.gn_pred_dim = 1

    # train_data, valid = data_split(code_mat, 0.1)
    # graph_train, reg_train = data_split(train_data, 0.4)

    # train = graph_train
    train, valid = data_split(code_mat, 0.3)

    links = np.array(links)
    g = construct_reverse_graph_from_edges(links.T, num_nodes, num_rel=4)
    g.ndata['feat'] = torch.ones([g.num_nodes(), 1], dtype=torch.float32)
    emb_pick = prepare_ct_graph(c2d, g.num_nodes(), max_ct_len, padding_index, e2n, w2i)
    params.num_nodes = g.num_nodes()

    g.ndata['cont_text'] = torch.LongTensor(emb_pick)
    g = g.to(params.device)
    g.ndata['node_emb'] = torch.rand((g.num_nodes(), params.node_emb_dim), device = params.device, requires_grad=True)
    train_set = NodeTextDataset(train, g)
    valid_set = NodeTextDataset(valid, g)
    model = GraphTextModel(vocab_size, EMDEDDING_DIM, max_ft_len, max_ct_len,params).to(params.device)
    reg_model = [FCLayers(4, params.corr_dim, [128, 128, 128, 1]),FCLayers(4, params.corr_dim, [128, 128, 128, 1])]
    cca_learner = CCALearner('cca_train', train_set, model, {
            'total_cor':0,
            'total_graph_var':0,
            'total_ft_var':0,
            'total_loss':0
        }, Adam, reg_model=reg_model)
    cca_learner.setup_optimizer([{'lr':params.lr}])
    cotrain_task_learner = RegCotrainLearner('cotrain_task', train_set, model, {'mse1':0, 'mse2':0,}, Adam, reg_model=reg_model)
    cotrain_task_learner.setup_optimizer([ {'lr':params.lr}])
    eval_learner = GraphCotrainLearner('eval', valid_set, model, {
            'total_graph_var':0,
            'total_ft_var':0,
            'total_cor':0,
            'mse1':0,
            'mse2':0
        }, Adam, reg_model=reg_model)
    proj_learner = ProjectionLearner('proj', train_set, model, {}, Adam)
    if params.fixed_emb:
        cotrain_task_learner.load_model('gt_best.pth')
        trainer = RegCoTrainer(params.num_epoch, params)
        manager = Manager([[proj_learner, cotrain_task_learner], [eval_learner]], trainer, save_path = 'gt')
    else:
        trainer = ProjTrainer(params.num_epoch, params)
        manager = Manager([cca_learner,  [proj_learner, eval_learner]], trainer, save_path = 'gt')
    manager.setup()
    manager.train(trainer, metric_name='total_cor', device=params.device, save_model=params.save_model)
