import torch
import torch.nn as nn
from model.dgl.rgcn_model import RGCN
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import numpy as np
import dgl

from tqdm import tqdm


class Mem:

    def __init__(self):
        self.num_rels = 1
        self.aug_num_rels = 2
        self.rel_emb_dim = 64
        self.emb_dim = 64
        self.attn_rel_emb_dim = 64
        self.add_ht_emb = True
        self.has_attn = False
        self.num_gcn_layers = 4
        self.num_bases = 4
        self.dropout = 0
        self.edge_dropout = 0
        self.gnn_agg_type = 'sum'
        self.inp_dim = self.emb_dim*self.num_gcn_layers
        self.cbow_only = True
        self.model_name = 'indeed_train'
        self.retrain = True


def code_cleanup(code):
    return code[:2]+code[3:5]

def construct_reverse_graph_from_edges(edges, n_entities, dim=1):
    g = dgl.graph((np.concatenate((edges[0],edges[1])), np.concatenate((edges[1],edges[0]))), num_nodes=n_entities)
    g.edata['type'] = torch.tensor(np.zeros(2*len(edges[0])), dtype=torch.int32)
    g.ndata['feat'] = torch.ones([n_entities, dim], dtype=torch.float32)
    return g

def collate_func(samples):
    node_ids, contexts, targets = map(list, zip(*samples))
    
    return node_ids, contexts, targets

def move_batch_to_device(batch, device):
    nodes_ids, contexts, targets = batch
    nodes_ids_device = torch.tensor(nodes_ids, device=device)
    contexts_device = torch.tensor(contexts, device=device)
    targets_device = torch.tensor(targets, device=device)
    return nodes_ids_device, contexts_device, targets_device

class NodeTextDataset(Dataset):
    def __init__(self, data, graph):
        self.data = data
        self.num_sample = len(data)
        self.g = graph

    def __len__(self):
        return self.num_sample

    def __getitem__(self, index):
        node_id, context, target = self.data[index]
        return node_id, context, target


class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, params):
        super(CBOW, self).__init__()
        self.params = params
        self.gnn = RGCN(params)
        self.param_set_dim = self.params.emb_dim*self.params.num_gcn_layers
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        if self.params.cbow_only:
            self.linear1 = nn.Linear(embedding_dim, 128)
        else:
            self.linear1 = nn.Linear(embedding_dim+self.param_set_dim, 128)
        self.activation_function1 = nn.ReLU()
        
        self.linear2 = nn.Linear(128, vocab_size)
        

    def forward(self, data):
        g, context, node_ids = data
        if not self.params.cbow_only:
            self.graph_update(g)
        return self.mlp_update(g, context, node_ids)

    def freeze_graph(self):
        for p in self.gnn.parameters():
            p.requires_grad = False
        # self.gnn.train(False)

    def unfreeze_graph(self):
        for p in self.gnn.parameters():
            p.requires_grad = True
        # self.gnn.train(False)

    def graph_update(self, g):
        self.gnn(g)

    def mlp_update(self, g, context, node_ids):
        embeds = torch.sum(self.embeddings(context), 1)
        if self.params.cbow_only:
            out = embeds
            # print(torch.abs(self.linear1.weight).mean())
        else:
            node_repr = g.ndata['repr'][node_ids]
            out = torch.cat([embeds,
                        node_repr.view(-1, self.param_set_dim)], axis=1)
            # print(torch.abs(self.linear1.weight[:,:self.embedding_dim]).mean())
            # print(torch.abs(self.linear1.weight[:,-self.param_set_dim:]).mean())
        out = self.linear1(out)
        out = self.activation_function1(out)
        out = self.linear2(out)
        return out

    def get_word_emdedding(self, word):
        word = torch.tensor([word_to_ix[word]])
        return self.embeddings(word).view(1,-1)


params = Mem()
if params.cbow_only:
    params.model_file_name = params.model_name+"_only.pth"
else:
    params.model_file_name = params.model_name+".pth"
CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
EMDEDDING_DIM = 100


open_file = open('/home/research/jerry.kong/data/ICD-9-DATA.pkl', "rb")
pc_icd_9 = pkl.load(open_file)
open_file.close()

open_file = open('/home/research/jerry.kong/data/word-index-map.pkl', "rb")
[w2i, i2w] = pkl.load(open_file)
open_file.close()

open_file = open('/home/research/jerry.kong/data/e2n.pkl', "rb")
e2n = pkl.load(open_file)
open_file.close()

open_file = open('/home/research/jerry.kong/data/graph.pkl', "rb")
links = pkl.load(open_file)
open_file.close()

if torch.cuda.is_available():
    params.device = torch.device('cuda:0')
else:
    params.device = torch.device('cpu')

print("Read data")
oov_char = '^'
vocab_size = np.max(np.array(list(w2i.values())))
i2w[vocab_size] = oov_char
if vocab_size!= np.max(np.array(list(i2w.keys()))):
    print("dictionary mismatch")
    quit()
padding_char = ''
vocab_size+=1
w2i[padding_char]=vocab_size
i2w[vocab_size] = padding_char

padding_index = w2i[padding_char]
vocab_size += 1

num_nodes = np.max(np.array(list(e2n.values())))+1

print("Prepare vocab")

code_text = []


for icd, text in pc_icd_9:
    if len(text) == 0:
        continue
    clcd = code_cleanup(icd)
    node_idx = e2n[clcd]
    g = np.zeros((len(text), 1+2*CONTEXT_SIZE), dtype=int)
    w2i_lst = np.zeros(len(text)+2*CONTEXT_SIZE, dtype=int)+padding_index
    w2i_lst[CONTEXT_SIZE: -CONTEXT_SIZE] = [w2i[w] for w in text]
    g = np.lib.stride_tricks.sliding_window_view(w2i_lst, CONTEXT_SIZE*2+1)
    target = g[:, CONTEXT_SIZE]
    context = np.delete(g, CONTEXT_SIZE, 1)
    for i, r in enumerate(target):
        code_text.append([node_idx, context[i], r])

code_mat = np.array(code_text, dtype=object)
num_code_text = len(code_mat)
perm = np.random.permutation(num_code_text)
valid = int(num_code_text/10)
valid_code_text = code_mat[perm[:valid]]
train_code_text = code_mat[perm[valid:]]

model = CBOW(vocab_size, EMDEDDING_DIM, params).to(params.device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
start_ep = 0
if params.retrain:
    state_d = torch.load(params.model_file_name, map_location=params.device)
    optimizer.load_state_dict(state_d['optimizer'])
    model.load_state_dict(state_d['state_dict'])
    start_ep = state_d['epoch']
links = np.array(links)
g = construct_reverse_graph_from_edges(links.T, num_nodes, params.inp_dim)

train_set = NodeTextDataset(train_code_text, g)
valid_set = NodeTextDataset(valid_code_text, g)

#TRAINING
for epoch in range(start_ep, 50+start_ep):
    print("Epoch:",epoch)
    graph_update_epoch = (epoch-start_ep)%5==1
    total_loss = 0
    h1 = 0
    h5 = 0
    h10 = 0
    model.train()
    if graph_update_epoch:
        dataloader = DataLoader(train_set, batch_size=1024,  num_workers=32, pin_memory=True, prefetch_factor=2, shuffle=True, collate_fn=collate_func)
        model.unfreeze_graph()
    else:
        dataloader = DataLoader(train_set, batch_size=128,  num_workers=32, pin_memory=True, prefetch_factor=2, shuffle=True, collate_fn=collate_func)
        model.freeze_graph()
        t_g = train_set.g.to(params.device)
        model.graph_update(t_g)

    pbar = tqdm(dataloader)
    for batch in pbar:
        g_node, context, target = move_batch_to_device(batch, params.device)
        if graph_update_epoch:
            score = model((train_set.g.to(params.device), context, g_node))
        else:
            score = model.mlp_update(t_g, context, g_node)
        loss = loss_function(score, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total_loss+=loss.item()
            score = score.cpu().numpy()
            target = target.cpu().numpy()
            ranking = np.argsort(-score, axis=1)
            ranking = np.where(ranking == target[:, np.newaxis])[1]
            h1 += np.mean(ranking<1)
            h5 += np.mean(ranking<5)
            h10 += np.mean(ranking<10)
    print("h1:",h1/len(dataloader),"h5:",h5/len(dataloader),"h10:",h10/len(dataloader),"loss:",total_loss/len(dataloader))
    dataloader = DataLoader(valid_set, batch_size=128,  num_workers=32, pin_memory=True, prefetch_factor=2, shuffle=True, collate_fn=collate_func)
    pbar = tqdm(dataloader)
    h1 = 0
    h5 = 0
    h10 = 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        v_g = valid_set.g.to(params.device)
        model.graph_update(v_g)
        for batch in pbar:
            g_node, context, target = move_batch_to_device(batch, params.device)
            score = model.mlp_update(v_g, context, g_node)
            loss = loss_function(score, target)
            total_loss+=loss.item()
            score = score.cpu().numpy()
            target = target.cpu().numpy()
            ranking = np.argsort(-score, axis=1)
            ranking = np.where(ranking == target[:, np.newaxis])[1]
            h1 += np.mean(ranking<1)
            h5 += np.mean(ranking<5)
            h10 += np.mean(ranking<10)
    print("h1:",h1/len(dataloader),"h5:",h5/len(dataloader),"h10:",h10/len(dataloader),"loss:",total_loss/len(dataloader))
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, params.model_file_name)

