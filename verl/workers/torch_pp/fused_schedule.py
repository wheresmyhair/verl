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


_TYPE_PRIORITY = {"infer_forward": 0, "train_backward": 1, "train_forward": 2}


def _select_op(
    ready: List[Tuple[str, int]],
) -> Tuple[str, int]:
    """
    Pick the next op from a non-empty ready buffer.

    Priority rule (from the fused schedule algorithm):
      1. Smallest micro-batch index first.
      2. Tie-break by type: iF > tB > tF.
    """
    best = ready[0]
    for candidate in ready[1:]:
        c_mb, b_mb = candidate[1], best[1]
        if c_mb < b_mb or (
            c_mb == b_mb and _TYPE_PRIORITY[candidate[0]] < _TYPE_PRIORITY[best[0]]
        ):
            best = candidate
    return best


def build_default_fused_schedule(
    pp_size: int,
    num_micro_batches: int,
) -> Dict[int, List[str]]:
    """
    Generate a memory-safe fused schedule using the priority-based algorithm.

    Memory constraint: on each rank, ALL inference forward ops must complete
    BEFORE any training backward begins.  This allows inference weights to be
    offloaded before optimizer states are loaded, preventing peak-memory
    conflicts.  The constraint has zero makespan overhead (proven for all
    P, M configurations tested).

    Last rank (P-1) — first inference stage:
      Run all iF.0 .. iF.(M-1) first, then 1F1B for training.

    All other ranks — ready-buffer dispatch:
      Maintain a ready buffer of ops whose dependencies are satisfied.
      train_backward is excluded from the ready set while any infer_forward
      remains unscheduled (memory constraint).
      SelectOp: smallest micro-batch index first, tie-break iF > tB > tF.

    Returns:
        {0: ["tF.0", "tF.1", ...], 1: [...], ...}
    """
    M = num_micro_batches
    P = pp_size

    # ── Build the last-rank schedule directly ──
    last_rank_ops: List[str] = []
    for mb in range(M):
        last_rank_ops.append(repr(FusedScheduleOp("infer_forward", mb)))
    for mb in range(M):
        last_rank_ops.append(repr(FusedScheduleOp("train_forward", mb)))
        last_rank_ops.append(repr(FusedScheduleOp("train_backward", mb)))

    scheduled: Dict[int, List[str]] = {P - 1: last_rank_ops}
    completed: Set[Tuple[int, str, int]] = set()

    # Mark last-rank ops as completed (in schedule order)
    for token_str in last_rank_ops:
        op = parse_schedule([token_str])[0]
        completed.add((P - 1, op.op, op.micro_batch_id))

    if P == 1:
        return scheduled

    # ── Event-driven simulation for ranks 0..P-2 ──
    #
    # Each rank runs independently at its own pace.  An op is "ready"
    # when the dependency's completion_time <= the rank's current clock.
    # Each op takes 1 time unit.  When the ready buffer is empty the
    # rank advances its clock to the earliest dependency arrival.

    for r in range(P - 1):
        scheduled[r] = []

    remaining: Dict[int, Set[Tuple[str, int]]] = {}
    for r in range(P - 1):
        remaining[r] = set()
        for mb in range(M):
            remaining[r].add(("train_forward", mb))
            remaining[r].add(("infer_forward", mb))
            remaining[r].add(("train_backward", mb))

    # completion_time[(rank, op_type, mb)] = wall-clock time when op finishes
    completion_time: Dict[Tuple[int, str, int], int] = {}

    # Pre-populate last rank's completion times
    t = 0
    for token_str in last_rank_ops:
        op = parse_schedule([token_str])[0]
        t += 1
        completion_time[(P - 1, op.op, op.micro_batch_id)] = t

    def _dep_time(rank: int, op_type: str, mb: int) -> int:
        """Earliest time at which all dependencies for this op are met."""
        t_dep = 0
        if op_type == "train_forward":
            if rank > 0:
                key = (rank - 1, "train_forward", mb)
                if key in completion_time:
                    t_dep = max(t_dep, completion_time[key])
                else:
                    return -1  # dep not yet scheduled
            if rank == P - 1:
                key = (0, "infer_forward", mb)
                if key in completion_time:
                    t_dep = max(t_dep, completion_time[key])
                else:
                    return -1
        elif op_type == "train_backward":
            key_own = (rank, "train_forward", mb)
            if key_own not in completion_time:
                return -1
            t_dep = max(t_dep, completion_time[key_own])
            if rank < P - 1:
                key_next = (rank + 1, "train_backward", mb)
                if key_next not in completion_time:
                    return -1
                t_dep = max(t_dep, completion_time[key_next])
        elif op_type == "infer_forward" and rank < P - 1:
            key = (rank + 1, "infer_forward", mb)
            if key in completion_time:
                t_dep = max(t_dep, completion_time[key])
            else:
                return -1
        return t_dep

    clock: Dict[int, int] = {r: 0 for r in range(P - 1)}
    total_remaining = 3 * M * (P - 1)
    max_iterations = total_remaining * total_remaining

    for _ in range(max_iterations):
        if total_remaining <= 0:
            break

        # Find the rank with the earliest clock that has work to do
        progress = False
        for r in sorted(range(P - 1), key=lambda x: clock[x]):
            if not remaining[r]:
                continue

            # Memory constraint: no train_backward while infer_forward remains
            has_pending_iF = any(
                op == "infer_forward" for op, _ in remaining[r]
            )

            # Collect ready ops: dependency met by current clock
            ready = []
            earliest_future_dep = float("inf")
            for op_type, mb in remaining[r]:
                # Enforce memory constraint: block tB until all iF done
                if op_type == "train_backward" and has_pending_iF:
                    continue
                dt = _dep_time(r, op_type, mb)
                if dt < 0:
                    continue  # dep not scheduled yet
                if dt <= clock[r]:
                    ready.append((op_type, mb))
                elif dt < earliest_future_dep:
                    earliest_future_dep = dt

            if ready:
                op_type, mb = _select_op(ready)
                token = repr(FusedScheduleOp(op_type, mb))
                scheduled[r].append(token)
                clock[r] += 1
                completion_time[(r, op_type, mb)] = clock[r]
                remaining[r].discard((op_type, mb))
                total_remaining -= 1
                progress = True
            elif earliest_future_dep < float("inf"):
                # Advance clock to when next dep arrives
                clock[r] = earliest_future_dep
                progress = True

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
