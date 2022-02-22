import pickle as pkl
import numpy as np

def code_cleanup(code):
    return code[:2]+code[3:5]


def open_and_load_pickle(filename):
    open_file = open(filename, "rb")
    data = pkl.load(open_file)
    open_file.close()
    return data

def process_vocab(w2i, i2w, oov_char='^', padding_char=''):
    vocab_size = np.max(np.array(list(w2i.values())))
    i2w[vocab_size] = oov_char
    if vocab_size!= np.max(np.array(list(i2w.keys()))):
        print("dictionary mismatch")
        quit()
    vocab_size+=1
    w2i[padding_char]=vocab_size
    i2w[vocab_size] = padding_char

    padding_index = w2i[padding_char]
    vocab_size += 1
    return vocab_size, padding_index

def get_max_text_len(collection, ind):
    max_len = 0
    for data in collection:
        text = data[ind]
        if len(text)>max_len:
            max_len = len(text)
    return max_len


def get_indexed_data(data, max_prc_len, max_ft_len, max_ct_len, e2n, w2i, padding_index):
    code_text = []
    for icd, text, ane_type, los in data:
        if len(text) == 0:
            continue
        nodes = [e2n[code_cleanup(c)] for c in icd]
        node_indices = np.zeros(max_prc_len, dtype=int)-1
        node_indices[:len(nodes)] = nodes
        text_indices = np.zeros(max_ft_len, dtype=int)+padding_index
        text_indices[:len(text)] = [w2i[w] for w in text]
        code_text.append([node_indices, text_indices, ane_type, los])
    return code_text


def data_split(code_mat, ratio=0.1):
    num_code_text = len(code_mat)
    perm = np.random.permutation(num_code_text)
    valid = int(num_code_text*ratio)
    valid_code_text = code_mat[perm[:valid]]
    train_code_text = code_mat[perm[valid:]]
    return train_code_text, valid_code_text


def prepare_ct_graph(c2d, num_nodes, max_ct_len, padding_index, e2n, w2i):
    
    emb_pick = np.zeros((num_nodes, max_ct_len), dtype=int)+padding_index
    for k,v in c2d.items():
        clcd = code_cleanup(k)
        node_idx = e2n[clcd]
        emb_pick[node_idx,:len(v)] = [w2i[w] for w in v]
    return emb_pick

# for icd, text, icd_pc in pc_icd_9:
#     if len(text) == 0:
#         continue
#     clcd = code_cleanup(icd)
#     node_idx = e2n[clcd]
#     g = np.zeros((len(text), 1+2*CONTEXT_SIZE), dtype=int)
#     w2i_lst = np.zeros(len(text)+2*CONTEXT_SIZE, dtype=int)+padding_index
#     w2i_lst[CONTEXT_SIZE: -CONTEXT_SIZE] = [w2i[w] for w in text]
#     g = np.lib.stride_tricks.sliding_window_view(w2i_lst, CONTEXT_SIZE*2+1)
#     target = g[:, CONTEXT_SIZE]
#     # context = np.delete(g, CONTEXT_SIZE, 1)
#     for i, r in enumerate(target):
#         code_text.append([node_idx, g[i], r])