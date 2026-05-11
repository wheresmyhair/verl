"""
1F1B (one-forward-one-backward) pipeline schedule.

Generates the ordered list of operations each PP stage must execute.
Standard GPipe-style schedule with warmup -> steady state -> cooldown.

Schedule for PP=4, num_micro_batches=8:

    Stage 0:  F0 F1 F2 F3  F4B0 F5B1 F6B2 F7B3  B4 B5 B6 B7
    Stage 1:     F0 F1 F2  F3B0 F4B1 F5B2 F6B3  B4 B5 B6 B7
    Stage 2:        F0 F1  F2B0 F3B1 F4B2 F5B3  B4 B5 B6 B7
    Stage 3:           F0  F1B0 F2B1 F3B2 F4B3  B4 B5 B6 B7

Each operation is (op_type, micro_batch_id).
"""

from dataclasses import dataclass
from typing import List, Literal


@dataclass
class ScheduleOp:
    """A single pipeline operation."""

    op: Literal["forward", "backward"]
    micro_batch_id: int

    def __repr__(self) -> str:
        prefix = "F" if self.op == "forward" else "B"
        return f"{prefix}{self.micro_batch_id}"


def build_1f1b_schedule(
    pp_rank: int,
    pp_size: int,
    num_micro_batches: int,
) -> List[ScheduleOp]:
    """
    Build the 1F1B schedule for a given PP stage.

    Args:
        pp_rank: this stage's rank (0-indexed)
        pp_size: total number of PP stages
        num_micro_batches: M, total micro-batches in the batch

    Returns:
        Ordered list of ScheduleOp.

    Raises:
        ValueError: if num_micro_batches < pp_size (required for 1F1B)
    """
    if num_micro_batches < pp_size:
        raise ValueError(
            f"num_micro_batches ({num_micro_batches}) must be >= pp_size ({pp_size}). "
            f"With fewer micro-batches the pipeline cannot be filled."
        )

    schedule: List[ScheduleOp] = []
    num_warmup = pp_size - pp_rank - 1  # how many extra forwards before steady state
    num_steady = num_micro_batches - num_warmup
    fwd_id = 0  # next micro-batch to forward
    bwd_id = 0  # next micro-batch to backward

    # Warmup: only forwards
    for _ in range(num_warmup):
        schedule.append(ScheduleOp("forward", fwd_id))
        fwd_id += 1

    # Steady state: interleaved F then B
    for _ in range(num_steady):
        schedule.append(ScheduleOp("forward", fwd_id))
        fwd_id += 1
        schedule.append(ScheduleOp("backward", bwd_id))
        bwd_id += 1

    # Cooldown: only backwards
    for _ in range(num_warmup):
        schedule.append(ScheduleOp("backward", bwd_id))
        bwd_id += 1

    return schedule


def print_schedule(pp_size: int, num_micro_batches: int):
    """Pretty-print the full schedule (for debugging)."""
    for rank in range(pp_size):
        ops = build_1f1b_schedule(rank, pp_size, num_micro_batches)
        line = " ".join(repr(op) for op in ops)
        print(f"  Stage {rank}: {line}")
