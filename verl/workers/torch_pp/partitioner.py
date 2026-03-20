"""
Layer partitioning for pipeline parallelism.

Assigns transformer layers to PP stages.

Stage 0:            embed_tokens + layers[0..k]
Stages 1..P-2:      layers[k..m]
Stage P-1:          layers[m..N] + norm + lm_head
"""

from typing import List, Tuple

from transformers import AutoConfig


def compute_layer_assignment(
    num_layers: int,
    pp_size: int,
) -> List[Tuple[int, int]]:
    """
    Evenly distribute ``num_layers`` across ``pp_size`` stages.

    Returns list of (start, end) tuples — end is exclusive.
    Extra layers go to earlier stages.

    Example (28 layers, PP=4):
        [(0, 7), (7, 14), (14, 21), (21, 28)]
    """
    if pp_size == 1:
        return [(0, num_layers)]

    base = num_layers // pp_size
    extra = num_layers % pp_size

    assignments = []
    start = 0
    for i in range(pp_size):
        count = base + (1 if i < extra else 0)
        assignments.append((start, start + count))
        start += count

    return assignments


def get_num_layers(model_path: str, trust_remote_code: bool = True) -> int:
    """Read number of transformer layers from HuggingFace config."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(config, attr):
            return getattr(config, attr)
    raise ValueError(f"Cannot determine num_layers from {model_path}")


def get_hidden_size(model_path: str, trust_remote_code: bool = True) -> int:
    """Read hidden size from HuggingFace config."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    for attr in ("hidden_size", "n_embd", "d_model"):
        if hasattr(config, attr):
            return getattr(config, attr)
    raise ValueError(f"Cannot determine hidden_size from {model_path}")
