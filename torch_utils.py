import torch

def collate_func(samples):
    node_id, context, context_len, code_desc, ct_len, ane_type, los = map(list, zip(*samples))
    
    return node_id, context, context_len, code_desc, ct_len, ane_type, los

def move_batch_to_device(batch, device):
    node_id, context, context_len, code_desc, ct_len, ane_type, los = batch
    nodes_ids_device = torch.tensor(node_id, device=device)
    contexts_device = torch.tensor(context, device=device)
    contexts_len_device = torch.tensor(context_len, device=device)
    code_desc_device = torch.tensor(code_desc, device=device)
    ct_len_device = torch.tensor(ct_len, device=device)
    ane_type_device = torch.tensor(ane_type, device=device, dtype=torch.long)
    los_device = torch.tensor(los, device=device, dtype=torch.float)
    return nodes_ids_device, contexts_device, contexts_len_device, code_desc_device, ct_len_device, ane_type_device, los_device