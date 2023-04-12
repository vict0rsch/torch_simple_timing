import torch
import torch.distributed as dist


def initialized() -> bool:
    """
    Whether or not distributed training is initialized.
    ``False`` when not initialized or not available.

    Returns:
        bool: Distributed training is initialized.
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """
    Returns the number of processes in the current distributed training.
    Defaults to 1 in the case of non-distributed training.

    Returns:
        int: number of processes
    """
    return dist.get_world_size() if initialized() else 1


def synchronize() -> None:
    """
    Synchronizes:

    * nothing on CPU
    * per-GPU CUDA streams with :func:`torch.cuda.synchronize()`
    * across all processes in distributed training with
      :func:`torch.distributed.barrier()`

    """
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    if get_world_size() > 1:
        dist.barrier()
