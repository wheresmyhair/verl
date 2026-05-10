"""DualPipe-style NCCL P2P executor for rlpipe fused-forward.

Background
----------
DeepSeek's DualPipe (github.com/deepseek-ai/DualPipe) showed that
bidirectional pipeline-parallel P2P does NOT need gloo, NCCL non-blocking
communicators, NVSHMEM, or any exotic backend — stock PyTorch
`dist.batch_isend_irecv` is sufficient, IF the schedule is structured
so every rank, at every commit point, submits ALL of its concurrent
P2P traffic in a single batch_isend_irecv call.

Our 2026-05-08~09 NCCL deadlock exploration (smoke_m1..m6) failed
because rank r3 was issuing iF recvs alone while peer r2 was busy in
training compute and hadn't submitted any P2P yet. NCCL had nothing
to match. DualPipe avoids this by making every rank enter the same
"commit point" simultaneously with the full set of ops it needs at
that step. Within one batch_isend_irecv group, NCCL can match all
(send, recv) pairs across ranks; the deadlock requires asymmetric
group composition, which the DualPipe schedule guarantees never
happens.

This module provides:
1. `build_fused_schedule(P, M)` — per-rank list of `Step` describing
   what P2P + compute to do at each step.
2. `MegatronDualPipeExecutor` — runs the schedule by, at each step,
   building the comm_ops list, batch_isend_irecv'ing it, waiting,
   then doing the compute. Loosely modeled on `dualpipe.py`'s
   `_commit_and_wait_comm` + state machine.

For now we cover:
- Reverse-PP iF flow (rank P-1 → 0)
- Optional: tF/tB training overlay (placed after iF burst slots; the
  schedule generator decides exact placement)

Out of scope here: training fF/bB micro-batches that depend on
gradient accumulation; we delegate full training schedule to stock
Megatron and only handle the iF + harvest log_probs, with potential
overlap inside r3's warmup bubble. The full fused schedule (iF
interleaved into 1F1B steady) is a follow-up.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import logging
import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


@dataclass
class P2PSpec:
    """One P2P op to issue at a commit point."""
    direction: str  # "send" or "recv"
    peer_rank: int  # global rank in actor_pp_group
    tag: int  # match tag (0 = iF, 1 = tF, 2 = tB; per-rank-pair uniqueness)
    tensor: torch.Tensor  # buffer to send (send) or fill (recv); MUST be pre-allocated and on device
    micro_batch_id: int  # for bookkeeping


@dataclass
class Step:
    """One step on a rank: P2P ops to issue (sends + recvs), then compute."""
    sends: List[P2PSpec] = field(default_factory=list)
    recvs: List[P2PSpec] = field(default_factory=list)
    compute_op: Optional[str] = None  # "iF.k" | "tF.k" | "tB.k" | None (= idle)
    micro_batch_id: int = -1


def build_iF_only_schedule(pp_size: int, num_microbatches: int) -> List[List[str]]:
    """Simplest schedule: pure reverse-PP iF, no training overlay.

    Returns `sched[rank]` = list of compute-op strings (or "" for idle).

    iF flows: rank P-1 → P-2 → ... → 0. Each iF.k takes P-1 hops.
    Total length = M + (P-1) steps. iF.k starts at r=P-1 step k,
    arrives at r=0 step k + (P-1).

    Layout (P=4, M=4):
        step:     0    1    2    3    4    5    6
        r0:                      iF.0 iF.1 iF.2 iF.3
        r1:                 iF.0 iF.1 iF.2 iF.3
        r2:            iF.0 iF.1 iF.2 iF.3
        r3:       iF.0 iF.1 iF.2 iF.3
    """
    P = pp_size
    M = num_microbatches
    n_steps = M + (P - 1)
    sched = [[""] * n_steps for _ in range(P)]

    for k in range(M):
        for r in range(P):
            # rev_pp_rank = P-1-r; iF.k arrives at rev_pp r' at step k + r'
            # rank with rev_pp r' = P-1-r is global rank r
            rev_pp = P - 1 - r
            arrival_step = k + rev_pp
            if 0 <= arrival_step < n_steps:
                sched[r][arrival_step] = f"iF.{k}"
    return sched


def schedule_to_steps(
    sched: List[str],
    rank: int,
    pp_size: int,
    pp_world_ranks: List[int],
    *,
    iF_hidden_shape: Tuple[int, ...],
    iF_dtype: torch.dtype,
    device: torch.device,
) -> List[Step]:
    """Convert a per-rank op-string sequence into Step objects with
    explicit P2P specs and pre-allocated buffers.

    Reverse-PP iF flow rules:
      - rank r is "first_inf" iff (P-1-r) == 0, i.e. r == P-1
        first_inf produces hidden from input_ids (no recv); sends to r-1
      - rank r is "last_inf" iff (P-1-r) == P-1, i.e. r == 0
        last_inf receives hidden, applies lm_head (no send)
      - middle: recv from r+1, compute, send to r-1
    """
    P = pp_size
    rev_pp = P - 1 - rank
    is_first_inf = (rev_pp == 0)
    is_last_inf = (rev_pp == P - 1)

    # Peers in actor pp_group
    next_inf_world = pp_world_ranks[rank - 1] if rank - 1 >= 0 else None  # rev_pp + 1
    prev_inf_world = pp_world_ranks[rank + 1] if rank + 1 < P else None   # rev_pp - 1

    # We pre-allocate one recv buffer per (mb_id, role). For iF-only
    # schedule, only one role per mb_id, so a dict suffices.
    iF_recv_bufs: Dict[int, torch.Tensor] = {}
    iF_send_bufs: Dict[int, torch.Tensor] = {}

    n_steps = len(sched)
    steps: List[Step] = []

    for s in range(n_steps):
        step = Step()
        op = sched[s]
        if op.startswith("iF."):
            mb = int(op.split(".")[1])
            step.compute_op = op
            step.micro_batch_id = mb

            # Pre-emptive: at THIS commit point, do we receive iF.k input?
            # Yes if not first_inf (we need hidden from prev_inf = r+1)
            if not is_first_inf:
                buf = torch.empty(iF_hidden_shape, dtype=iF_dtype, device=device)
                iF_recv_bufs[mb] = buf
                step.recvs.append(P2PSpec(
                    direction="recv",
                    peer_rank=prev_inf_world,
                    tag=mb,  # tag = micro_batch id
                    tensor=buf,
                    micro_batch_id=mb,
                ))

            # Will we send iF.k output to next_inf (r-1)?
            # Yes if not last_inf. The send tensor is the compute output;
            # we mark a placeholder here, attached after compute.
            # NOTE: in DualPipe, send is co-batched with recv at the
            # SAME step where compute completes. We move the send to
            # the NEXT step's batch (since the tensor isn't ready yet).
            # But that requires forward-look. Simpler: in this version,
            # we co-batch send-of-step-s with recv-of-step-s+1 in step
            # s+1's batch. So step `s` sends are deferred to step s+1.
            # This makes step boundaries: pre-compute sends + pre-compute recvs.
            # For first iteration, there's no prior compute to send from,
            # so step 0's sends list is empty.
        steps.append(step)

    # Second pass: attach sends. The send for compute at step s, on a
    # rank that is not last_inf, happens BEFORE the next step's compute,
    # i.e. it is co-batched with step s+1's recv (so the next-rank can
    # match it in its step s+1 batch_isend_irecv).
    if not is_last_inf:
        for s in range(n_steps - 1):
            op = sched[s]
            if op.startswith("iF."):
                mb = int(op.split(".")[1])
                # The send tensor will be the result of compute_op at step s.
                # We allocate the buffer here as a placeholder; the executor
                # writes the compute output into it before the next step's batch.
                buf = torch.empty(iF_hidden_shape, dtype=iF_dtype, device=device)
                iF_send_bufs[mb] = buf
                steps[s + 1].sends.append(P2PSpec(
                    direction="send",
                    peer_rank=next_inf_world,
                    tag=mb,
                    tensor=buf,
                    micro_batch_id=mb,
                ))

    return steps, iF_send_bufs, iF_recv_bufs


class MegatronDualPipeExecutor:
    """Run a fused schedule using DualPipe-style co-batched NCCL P2P.

    Caller pre-allocates buffers and provides:
      - infer_compute_func(input_ids, position_ids, attention_mask, hidden_in=None) → hidden_out
        (last_inf returns log_probs instead of hidden)
      - data: dict with input_ids, position_ids, attention_mask split by mb

    NCCL P2P discipline:
      - At every step, build comm_ops = sends + recvs (in order)
      - dist.batch_isend_irecv(comm_ops); reqs[*].wait()
      - Then do local compute (if any)
      - This makes the "commit point" a global barrier where all ranks'
        groups have matching peers. No more 30-min watchdog deadlocks.
    """

    def __init__(
        self,
        *,
        pp_size: int,
        rank: int,
        pp_group,
        pp_world_ranks: List[int],
    ):
        self.pp_size = pp_size
        self.rank = rank
        self.pp_group = pp_group
        self.pp_world_ranks = pp_world_ranks
        self.rev_pp = pp_size - 1 - rank
        self.is_first_inf = (self.rev_pp == 0)
        self.is_last_inf = (self.rev_pp == pp_size - 1)

    def run_iF(
        self,
        steps: List[Step],
        *,
        infer_compute_func: Callable,
        iF_send_bufs: Dict[int, torch.Tensor],
        iF_recv_bufs: Dict[int, torch.Tensor],
        data_per_mb: List[Dict[str, torch.Tensor]],  # mb_id -> {input_ids, position_ids, attention_mask}
    ) -> Dict[int, torch.Tensor]:
        """Execute iF-only schedule. Returns log_probs per micro-batch
        (only on last_inf rank; other ranks get an empty dict)."""
        log_probs: Dict[int, torch.Tensor] = {}

        for s, step in enumerate(steps):
            # 1. Build comm_ops for this step
            comm_ops = []
            for spec in step.sends:
                comm_ops.append(dist.P2POp(
                    op=dist.isend,
                    tensor=spec.tensor,
                    peer=spec.peer_rank,
                    group=self.pp_group,
                    tag=spec.tag,
                ))
            for spec in step.recvs:
                comm_ops.append(dist.P2POp(
                    op=dist.irecv,
                    tensor=spec.tensor,
                    peer=spec.peer_rank,
                    group=self.pp_group,
                    tag=spec.tag,
                ))

            # 2. Submit + wait — DualPipe's _commit_and_wait_comm
            if comm_ops:
                reqs = dist.batch_isend_irecv(comm_ops)
                for req in reqs:
                    req.wait()

            # 3. Local compute (if any)
            if step.compute_op is None:
                continue
            if step.compute_op.startswith("iF."):
                mb = step.micro_batch_id
                hidden_in = iF_recv_bufs.get(mb)  # None if first_inf
                data_mb = data_per_mb[mb]
                output = infer_compute_func(
                    input_ids=data_mb["input_ids"],
                    position_ids=data_mb["position_ids"],
                    attention_mask=data_mb["attention_mask"],
                    hidden_in=hidden_in,
                    is_first_inf=self.is_first_inf,
                    is_last_inf=self.is_last_inf,
                )
                if self.is_last_inf:
                    # output is log_probs
                    log_probs[mb] = output
                else:
                    # output is hidden state; copy into pre-allocated send buffer
                    if mb in iF_send_bufs:
                        iF_send_bufs[mb].copy_(output)

        return log_probs


def compute_log_prob_reverse_pp(
    *,
    infer_module,
    micro_batches: list,  # list[dict] each with input_ids, attention_mask, position_ids, responses
    pp_group,
    temperature: float = 1.0,
    pad_to_seq_len: int = None,
):
    """Reverse-PP compute_log_prob via DualPipe-style NCCL P2P.

    Each micro-batch flows rank P-1 → ... → 0 through `infer_module`.
    Last_inf rank (0) computes log_probs from logits + responses; we
    broadcast the result to all PP ranks for downstream use.

    Per-micro-batch protocol:
      - boundary k (between iF.k-1 and iF.k):
          * recv iF.k input hidden (if not first_inf)
          * send iF.k-1 output hidden (if not last_inf and k>0)
      - compute iF.k via infer_module(input_ids, ...)
      - last_inf: extract log_probs[response slice] from logits
    Final drain: send iF.M-1 output (if not last_inf).

    NOTE: This routes the forward through `infer_module` which has
    REVERSED layer assignment + correct pre/post_process flags from
    `build_reverse_pp_inference_model`. The model expects
    set_input_tensor(recv_hidden) for non-pre_process stages.

    Returns: log_probs tensor of shape (total_B, response_length) on
             every PP rank (broadcast from last_inf).
    """
    pp_size = dist.get_world_size(pp_group)
    pp_rank = dist.get_rank(pp_group)
    pp_world_ranks = dist.get_process_group_ranks(pp_group)
    rev_pp = pp_size - 1 - pp_rank
    is_first_inf = (rev_pp == 0)
    is_last_inf = (rev_pp == pp_size - 1)

    prev_world = pp_world_ranks[pp_rank + 1] if pp_rank + 1 < pp_size else None
    next_world = pp_world_ranks[pp_rank - 1] if pp_rank - 1 >= 0 else None

    M = len(micro_batches)
    if M == 0:
        return torch.zeros(0)

    # Probe hidden shape from first_inf (rank P-1) for the recv buffer
    # allocation. We use input shape & infer_module config to derive.
    sample_ids = micro_batches[0]["input_ids"]
    bsz = sample_ids.shape[0]
    seq_len = pad_to_seq_len if pad_to_seq_len else sample_ids.shape[1]
    cfg = infer_module.config
    hidden_size = cfg.hidden_size
    # Megatron expects (seq, batch, hidden) for PP transfer with seq-first
    # layout (variable_seq_lengths ON). For pack_seqs=False / no padding
    # path it's (batch, seq, hidden). We use seq-first as Megatron default.
    hidden_shape = (seq_len, bsz, hidden_size)
    dtype = next(infer_module.parameters()).dtype
    device = next(infer_module.parameters()).device

    TAG_iF = lambda mb: 30000 + mb  # large base to avoid collision

    log_probs_per_mb = {}
    last_compute_hidden = None  # output of iF.k-1, kept on GPU for next boundary's send

    for k in range(M):
        comm_ops = []
        recv_buf = None
        if not is_first_inf:
            recv_buf = torch.empty(hidden_shape, dtype=dtype, device=device)
            comm_ops.append(dist.P2POp(
                op=dist.irecv,
                tensor=recv_buf,
                peer=prev_world,
                group=pp_group,
                tag=TAG_iF(k),
            ))
        if k > 0 and not is_last_inf and last_compute_hidden is not None:
            comm_ops.append(dist.P2POp(
                op=dist.isend,
                tensor=last_compute_hidden,
                peer=next_world,
                group=pp_group,
                tag=TAG_iF(k - 1),
            ))

        if comm_ops:
            reqs = dist.batch_isend_irecv(comm_ops)
            for req in reqs:
                req.wait()

        # Compute iF.k
        mb = micro_batches[k]
        with torch.no_grad():
            if recv_buf is not None:
                infer_module.set_input_tensor(recv_buf)
            # else: first_inf uses embedding(input_ids) internally; don't
            # set input_tensor (None would mislead the decoder).

            output = infer_module(
                input_ids=mb["input_ids"],
                position_ids=mb["position_ids"],
                attention_mask=mb["attention_mask"],
            )

        if is_last_inf:
            # output is logits of shape (batch, seq, vocab_partition).
            # Apply temperature and compute log_probs over responses.
            from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits
            logits = output / temperature
            responses = mb["responses"]
            # Build labels aligned with input_ids; last_response_len-1..-1 are response tokens
            # log_prob[i] = log P(response_i | prefix_<= i-1)
            response_length = responses.size(1)
            position_ids = mb["position_ids"]
            label = position_ids.clone()
            label[:, -response_length - 1 : -1] = responses
            log_probs = vocab_parallel_log_probs_from_logits(logits, label)
            # Slice to response positions only
            log_probs = log_probs[:, -response_length - 1 : -1].contiguous()
            log_probs_per_mb[k] = log_probs
            last_compute_hidden = None
        else:
            # output is hidden state to forward
            last_compute_hidden = output.contiguous()

    # Final drain
    if not is_last_inf and last_compute_hidden is not None:
        reqs = dist.batch_isend_irecv([dist.P2POp(
            op=dist.isend,
            tensor=last_compute_hidden,
            peer=next_world,
            group=pp_group,
            tag=TAG_iF(M - 1),
        )])
        for req in reqs:
            req.wait()

    # Aggregate log_probs across micro-batches on last_inf, then broadcast
    last_inf_world = pp_world_ranks[pp_size - 1 - (pp_size - 1)]  # = world rank corresponding to rev_pp=P-1 = global rank 0
    if is_last_inf:
        all_log_probs = torch.cat([log_probs_per_mb[k] for k in range(M)], dim=0).to(torch.float32)
    else:
        # Allocate matching shape based on first mb's shape × M
        total_B = sum(mb["responses"].shape[0] for mb in micro_batches)
        response_length = micro_batches[0]["responses"].shape[1]
        all_log_probs = torch.empty((total_B, response_length), dtype=torch.float32, device=device)

    dist.broadcast(all_log_probs, src=last_inf_world, group=pp_group)
    return all_log_probs
