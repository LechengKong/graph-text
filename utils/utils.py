import time
import numpy as np
from scipy.sparse import csr_matrix, tril
import scipy.io as io

class SmartTimer():
    def __init__(self, verb = True) -> None:
        self.last = time.time()
        self.verb = verb

    def record(self):
        self.last = time.time()
    
    def cal_and_update(self, name):
        now = time.time()
        if self.verb:
            print(name,now-self.last)
        self.record()


def get_rank(b_score):
    order = np.argsort(b_score)
    return len(order)-np.where(order==0)[0][0]

def read_knowledge_graph(files, relation2id=None):
    entity2id = {}
    if relation2id is None:
        relation2id = {}

    converted_triplets = {}
    rel_list = [[] for i in range(len(relation2id))]

    ent = 0
    rel = len(relation2id)

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1
                rel_list.append([])

            data.append([entity2id[triplet[0]], relation2id[triplet[1]], entity2id[triplet[2]]])
        
        for trip in data:
            rel_list[trip[1]].append([trip[0], trip[2]])

        converted_triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    adj_list = []
    for rel_mat in rel_list:
        rel_array = np.array(rel_mat)
        if len(rel_array)==0:
            adj_list.append(csr_matrix((len(entity2id),len(entity2id))))
        else:
            adj_list.append(csr_matrix((np.ones(len(rel_mat)),(rel_array[:,0],rel_array[:,1])), shape=(len(entity2id),len(entity2id))))

    return adj_list, converted_triplets, entity2id, relation2id, id2entity, id2relation

def read_homogeneous_graph(path, is_text=False):
    if is_text:
        data = np.genfromtxt(path, delimiter=',')
        train_num_entities = np.max(data)+1
        ind_num_entities = train_num_entities
        head, tail = data[:,0], data[:,1]
    else:
        Amat = io.loadmat(path)['net']
        train_num_entities = Amat.shape[0]
        ind_num_entities = train_num_entities
        edge_mat = tril(Amat)
        head, tail = edge_mat.nonzero()
    train_num_rel = 1
    ind_num_rel = 1
    relation2id = None
    k = np.ones((train_num_entities,train_num_entities))
    k[head,tail]=0
    k[tail,head]=0
    nh,nt = k.nonzero()
    neg_perm = np.random.permutation(len(nt))
    perm = np.random.permutation(len(head))
    train_ind = int(len(perm)*0.85)
    test_ind = int(len(perm)*0.95)
    new_mat = np.zeros((len(head),3),dtype=int)
    new_mat[:,0] = head
    new_mat[:,2] = tail
    neg_mat = np.zeros((len(head),3), dtype=int)
    neg_mat[:,0] = nh[neg_perm[:len(head)]]
    neg_mat[:,2] = nt[neg_perm[:len(head)]]
    converted_triplets = {"train":new_mat[perm[:train_ind]], "train_neg":neg_mat[perm[:train_ind]], "test":new_mat[perm[train_ind:test_ind]],"test_neg":neg_mat[perm[train_ind:test_ind]], "valid":new_mat[perm[test_ind:]], "valid_neg":neg_mat[perm[test_ind:]]}
    converted_triplets_ind = converted_triplets
    rel_mat = converted_triplets['train']
    adj_list = [csr_matrix((np.ones(len(rel_mat)),(rel_mat[:,0],rel_mat[:,1])), shape=(train_num_entities,train_num_entities))]
    adj_list_ind = adj_list
    return adj_list, converted_triplets, relation2id

def save_params(filename, params):
    with open(filename, 'a') as f:
        f.write("\n\n")
        d = vars(params)
        string = "python train_graph.py "
        for k in d:
            string+="--"+k+" "+str(d[k])+" "
        f.write(string)


    # for row in data:
    #     if row[0] not in entity2id:
    #         entity2id[row[0]] = ent
    #         ent += 1
    #     if row[1] not in entity2id:
    #         entity2id[row[1]] = ent
    #         ent += 1
    #     edges.append([entity2id[row[0]], entity2id[row[1]]])