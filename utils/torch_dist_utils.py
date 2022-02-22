import torch.distributed as dist


def init_process_group(world_size, rank):
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12345',
        world_size=world_size,
        rank=rank)

