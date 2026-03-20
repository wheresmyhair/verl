"""
Fused forward schedule — user-provided operation lists per GPU rank.

Training flows GPU0->GPU3 (forward direction); inference flows GPU3->GPU0
(reverse direction).  Each GPU holds both a training stage and an
inference stage (reversed layer assignment).

Schedule format per rank: list of string tokens
    "tF.0"  -> train_forward  micro-batch 0
    "tB.2"  -> train_backward micro-batch 2
    "iF.1"  -> infer_forward  micro-batch 1

The worker receives a parsed ``List[FusedScheduleOp]`` and iterates it
sequentially.  Each op implies a specific recv source / send destination
based on op type and the GPU's rank.
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Set, Tuple


@dataclass
class FusedScheduleOp:
    """A single fused-pipeline operation."""

    op: Literal["train_forward", "train_backward", "infer_forward"]
    micro_batch_id: int

    def __repr__(self) -> str:
        prefixes = {
            "train_forward": "tF",
            "train_backward": "tB",
            "infer_forward": "iF",
        }
        return f"{prefixes[self.op]}.{self.micro_batch_id}"


# -----------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------

_TOKEN_TO_OP = {
    "tF": "train_forward",
    "tB": "train_backward",
    "iF": "infer_forward",
}


def parse_schedule(ops_str: List[str]) -> List[FusedScheduleOp]:
    """
    Parse ``["tF.0", "tB.1", "iF.2", ...]`` into ``FusedScheduleOp`` list.

    Raises:
        ValueError: on malformed tokens.
    """
    result: List[FusedScheduleOp] = []
    for token in ops_str:
        parts = token.split(".")
        if len(parts) != 2 or parts[0] not in _TOKEN_TO_OP:
            raise ValueError(
                f"Invalid schedule token '{token}'. "
                f"Expected format: tF.<int>, tB.<int>, or iF.<int>"
            )
        try:
            mb = int(parts[1])
        except ValueError:
            raise ValueError(
                f"Invalid micro-batch id in '{token}'. Expected integer after '.'"
            )
        result.append(FusedScheduleOp(_TOKEN_TO_OP[parts[0]], mb))
    return result


# -----------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------


def validate_schedule(
    schedules: Dict[int, List[FusedScheduleOp]],
    pp_size: int,
    num_micro_batches: int,
) -> None:
    """
    Validate user-provided schedules.

    Checks:
        1. Each rank has exactly M train_forward, M train_backward, M infer_forward
        2. Data dependencies: no op before its input is produced by the source rank
        3. No deadlocks (circular waits) — checked via topological ordering

    Raises:
        ValueError: with details on any violation.
    """
    M = num_micro_batches
    P = pp_size

    if set(schedules.keys()) != set(range(P)):
        raise ValueError(
            f"Schedule must have entries for ranks 0..{P-1}. "
            f"Got ranks: {sorted(schedules.keys())}"
        )

    # Check 1: Op counts
    for rank, ops in schedules.items():
        counts = {"train_forward": 0, "train_backward": 0, "infer_forward": 0}
        for op in ops:
            counts[op.op] += 1
        for op_type, count in counts.items():
            if count != M:
                raise ValueError(
                    f"Rank {rank}: expected {M} {op_type} ops, got {count}"
                )

    # Build position maps: (rank, op_type, mb) -> position index in that rank's list
    pos: Dict[Tuple[int, str, int], int] = {}
    for rank, ops in schedules.items():
        for i, op in enumerate(ops):
            pos[(rank, op.op, op.micro_batch_id)] = i

    # Check 2: Data dependencies
    errors: List[str] = []
    for rank, ops in schedules.items():
        for i, op in enumerate(ops):
            mb = op.micro_batch_id

            if op.op == "train_forward":
                if rank > 0:
                    src_key = (rank - 1, "train_forward", mb)
                    if src_key not in pos:
                        errors.append(
                            f"tF.{mb} @ rank {rank}: missing tF.{mb} at rank {rank-1}"
                        )
                if rank == P - 1:
                    olp_key = (0, "infer_forward", mb)
                    if olp_key not in pos:
                        errors.append(
                            f"tF.{mb} @ rank {rank}: missing iF.{mb} at rank 0 "
                            f"for old_log_probs"
                        )
                    elif rank == 0:  # PP=1: same-rank, check position
                        if pos[olp_key] >= i:
                            errors.append(
                                f"tF.{mb} @ rank {rank}: iF.{mb} must come before "
                                f"tF.{mb} for old_log_probs "
                                f"(iF at pos {pos[olp_key]}, tF at pos {i})"
                            )

            elif op.op == "train_backward":
                own_fwd_key = (rank, "train_forward", mb)
                if own_fwd_key not in pos:
                    errors.append(
                        f"tB.{mb} @ rank {rank}: missing tF.{mb} on same rank"
                    )
                elif pos[own_fwd_key] >= i:
                    errors.append(
                        f"tB.{mb} @ rank {rank}: tF.{mb} must come before tB.{mb} "
                        f"(tF at pos {pos[own_fwd_key]}, tB at pos {i})"
                    )
                if rank < P - 1:
                    src_key = (rank + 1, "train_backward", mb)
                    if src_key not in pos:
                        errors.append(
                            f"tB.{mb} @ rank {rank}: missing tB.{mb} at rank {rank+1}"
                        )

            elif op.op == "infer_forward" and rank < P - 1:
                src_key = (rank + 1, "infer_forward", mb)
                if src_key not in pos:
                    errors.append(
                        f"iF.{mb} @ rank {rank}: missing iF.{mb} at rank {rank+1}"
                    )

    if errors:
        raise ValueError(
            "Schedule dependency violations:\n  " + "\n  ".join(errors)
        )

    # Check 3: Deadlock detection via topological sort
    edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    in_degree: Dict[Tuple[int, int], int] = {}

    for rank, ops in schedules.items():
        for i in range(len(ops)):
            in_degree[(rank, i)] = 0

    # Sequential dependencies within each rank
    for rank, ops in schedules.items():
        for i in range(len(ops) - 1):
            edges.append(((rank, i), (rank, i + 1)))
            in_degree[(rank, i + 1)] += 1

    # Cross-rank dependencies
    for rank, ops in schedules.items():
        for i, op in enumerate(ops):
            mb = op.micro_batch_id
            if op.op == "train_forward":
                if rank > 0:
                    src_pos = pos[(rank - 1, "train_forward", mb)]
                    edges.append(((rank - 1, src_pos), (rank, i)))
                    in_degree[(rank, i)] += 1
                if rank == P - 1:
                    src_pos = pos[(0, "infer_forward", mb)]
                    edges.append(((0, src_pos), (rank, i)))
                    in_degree[(rank, i)] += 1
            elif op.op == "train_backward" and rank < P - 1:
                src_pos = pos[(rank + 1, "train_backward", mb)]
                edges.append(((rank + 1, src_pos), (rank, i)))
                in_degree[(rank, i)] += 1
            elif op.op == "infer_forward" and rank < P - 1:
                src_pos = pos[(rank + 1, "infer_forward", mb)]
                edges.append(((rank + 1, src_pos), (rank, i)))
                in_degree[(rank, i)] += 1

    # Kahn's algorithm
    queue = [node for node, deg in in_degree.items() if deg == 0]
    visited = 0
    adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = {
        node: [] for node in in_degree
    }
    for src, dst in edges:
        adj[src].append(dst)

    while queue:
        node = queue.pop(0)
        visited += 1
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    total_nodes = sum(len(ops) for ops in schedules.values())
    if visited != total_nodes:
        raise ValueError(
            f"Schedule contains a deadlock (cycle detected). "
            f"Topological sort visited {visited}/{total_nodes} nodes."
        )


# -----------------------------------------------------------------------
# Default schedule generator
# -----------------------------------------------------------------------


def build_default_fused_schedule(
    pp_size: int,
    num_micro_batches: int,
) -> Dict[int, List[str]]:
    """
    Generate a default fused schedule (greedy simulation) as string lists.

    Strategy: simulate time steps assuming uniform op duration.  Each rank
    greedily picks the next ready op from its queue (train_forward first,
    then infer_forward, then train_backward) checking that the data
    dependency from the source rank has been scheduled in an earlier slot.

    Returns:
        {0: ["tF.0", "tF.1", ...], 1: [...], ...}
    """
    M = num_micro_batches
    P = pp_size

    scheduled: Dict[int, List[str]] = {r: [] for r in range(P)}
    completed: Set[Tuple[int, str, int]] = set()

    pending: Dict[int, List[Tuple[str, int]]] = {r: [] for r in range(P)}
    for r in range(P):
        for mb in range(M):
            pending[r].append(("train_forward", mb))
        for mb in range(M):
            pending[r].append(("infer_forward", mb))
        for mb in range(M):
            pending[r].append(("train_backward", mb))

    def _is_ready(rank: int, op_type: str, mb: int) -> bool:
        if op_type == "train_forward":
            if rank > 0 and (rank - 1, "train_forward", mb) not in completed:
                return False
            if rank == P - 1 and (0, "infer_forward", mb) not in completed:
                return False
            return True
        if op_type == "train_backward":
            if (rank, "train_forward", mb) not in completed:
                return False
            if rank < P - 1:
                return (rank + 1, "train_backward", mb) in completed
            return True
        if op_type == "infer_forward" and rank < P - 1:
            return (rank + 1, "infer_forward", mb) in completed
        return True  # Last rank's iF: no dependency

    total_ops = 3 * M * P
    scheduled_count = 0
    max_iterations = total_ops * total_ops

    iteration = 0
    while scheduled_count < total_ops and iteration < max_iterations:
        iteration += 1
        progress = False
        for r in range(P):
            if not pending[r]:
                continue
            for idx, (op_type, mb) in enumerate(pending[r]):
                if _is_ready(r, op_type, mb):
                    token = repr(FusedScheduleOp(op_type, mb))
                    scheduled[r].append(token)
                    completed.add((r, op_type, mb))
                    pending[r].pop(idx)
                    scheduled_count += 1
                    progress = True
                    break
        if not progress:
            raise RuntimeError(
                "Default schedule generator stuck — cannot find any ready op. "
                "This should not happen for valid pp_size/num_micro_batches."
            )

    return scheduled


# -----------------------------------------------------------------------
# Pretty-print
# -----------------------------------------------------------------------


def print_fused_schedule(schedules: Dict[int, List[str]]):
    """Print the schedule in a readable table format."""
    max_ops = max(len(ops) for ops in schedules.values())
    header = "Step |" + "|".join(f" Rank {r:2d} " for r in sorted(schedules))
    print(header)
    print("-" * len(header))
    for step in range(max_ops):
        parts = []
        for r in sorted(schedules):
            ops = schedules[r]
            if step < len(ops):
                parts.append(f" {ops[step]:>6s} ")
            else:
                parts.append("        ")
        print(f"{step:4d} |" + "|".join(parts))
