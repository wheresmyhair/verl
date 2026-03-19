import torch

def save_log_by_rank(log_content: str):
    """
    Save log by rank.
    """
    rank = torch.distributed.get_rank()
    with open(f"./log_rank_{rank}.txt", "a") as f:
        f.write(f"{log_content}\n")