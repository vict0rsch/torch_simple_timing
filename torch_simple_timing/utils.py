import torch
import torch.distributed as dist


def initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if initialized() else 1


def synchronize():
    if get_world_size() > 1:
        dist.barrier()
    else:
        torch.cuda.synchronize()
