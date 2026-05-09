"""M8 — FusedScheduleExecutor end-to-end smoke (toy PP model).

Validates the schedule loop: build_default_fused_schedule → executor →
gloo P2P → tF/tB/iF dispatch. Uses a TOY PP model (not Megatron) so we
can run it on 4 GPUs without full Megatron init.

Toy model
---------
Each rank holds a single nn.Linear "layer". tF takes input [S, B, H],
applies Linear, sends to next. tB receives grad, backprops Linear,
sends grad to prev. Last rank computes scalar "loss" = output.sum()
and starts backward.

iF: in `infer_local=True` mode, each rank does a no-grad forward of its
Linear on a stashed input. Result discarded (smoke just checks the
schedule completes without deadlock).

Run:
  torchrun --nproc-per-node=4 \
    /home/user/rlpipe/verl/exp_script/megatron_fused/smoke_m8_executor.py
"""
from __future__ import annotations
import os, sys, time

import torch
import torch.nn as nn
import torch.distributed as dist

sys.path.insert(0, "/home/user/rlpipe/verl")


PP = 4
M = 8
SEQ = 512
HIDDEN = 1024
DTYPE = torch.float32  # use float32 to keep grads stable


class ToyLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden, bias=False)
        self._input_tensor = None  # set by set_input_tensor

    def set_input_tensor(self, input_tensor):
        # Megatron compat: list of one
        self._input_tensor = input_tensor[0] if isinstance(input_tensor, list) else input_tensor

    def forward(self, x):
        if x is None:
            x = self._input_tensor
        return self.linear(x)


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == PP
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    print(f"[r{rank}] init", flush=True)

    from verl.utils.megatron.fused_p2p import create_megatron_fused_pair_groups
    pair_groups = create_megatron_fused_pair_groups(PP, rank)
    print(f"[r{rank}] pair_groups: {sorted([str(k) for k in pair_groups.keys()])}", flush=True)

    # Build toy model on this rank
    device = torch.device("cuda")
    model = ToyLayer(HIDDEN).to(device).to(DTYPE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    # Pre-chunk synthetic data: M micro-batches of (SEQ, 1, HIDDEN)
    micro_batches = [torch.randn(SEQ, 1, HIDDEN, device=device, dtype=DTYPE) for _ in range(M)]

    # forward_step_func mimics megatron interface
    mb_counter = [0]

    def forward_step_func(data_iterator, model_inner):
        mb_id = next(data_iterator)  # iterator yields mb id
        if rank == 0:
            x = micro_batches[mb_id]
        else:
            x = None  # set_input_tensor will provide
        out = model_inner(x)
        # loss_func: only meaningful at last rank
        if rank == PP - 1:
            def loss_func(output_tensor):
                return output_tensor.sum() * 1e-6
            return out, loss_func
        return out, None

    def backward_step_func(*, input_tensor, output_tensor, output_tensor_grad,
                           model_type=None, config=None):
        """Mimics megatron's backward_step. Returns input_tensor_grad."""
        if output_tensor_grad is None:
            # last stage: output_tensor IS the loss (scalar)
            output_tensor.backward()
        else:
            output_tensor.backward(output_tensor_grad)
        return input_tensor.grad if input_tensor is not None and input_tensor.requires_grad else None

    # data_iterator: simple generator over [0..M-1] cycling per call
    def make_iter():
        for i in range(M):
            yield i

    data_iter = iter(make_iter())

    # Build schedule
    from verl.workers.torch_pp.fused_schedule import (
        build_default_fused_schedule, parse_schedule,
    )
    sched = build_default_fused_schedule(PP, M)
    schedule_ops = parse_schedule(sched[rank])
    print(f"[r{rank}] schedule: {[(op.op[0]+op.op[-1].upper(), op.micro_batch_id) for op in schedule_ops[:8]]}... (total {len(schedule_ops)})", flush=True)

    # Run via FusedScheduleExecutor
    from verl.utils.megatron.fused_schedule_executor import FusedScheduleExecutor

    # NOTE: This smoke uses a non-Megatron toy. Patch _call_megatron_*
    # to use our toy paths.
    class ToyExecutor(FusedScheduleExecutor):
        def _call_megatron_forward_step(self, input_tensor, mb):
            # Set input on model
            self.train_model.set_input_tensor([input_tensor] if input_tensor is not None else [None])
            # Get next mb id from our data_iter (cycles per fwd call)
            # We use mb directly here for simplicity
            if self.is_first:
                x = micro_batches[mb]
            else:
                x = None
            out = self.train_model(x)
            if self.is_last:
                # apply scalar loss for backward
                loss = out.sum() * 1e-6 / self.M
                return loss
            return out

        def _call_megatron_backward_step(self, input_tensor, output_tensor, output_tensor_grad):
            if output_tensor_grad is None:
                output_tensor.backward()
            else:
                output_tensor.backward(output_tensor_grad)
            return input_tensor.grad if (input_tensor is not None and input_tensor.requires_grad) else None

    dist.barrier()
    t_start = time.perf_counter()

    executor = ToyExecutor(
        train_model=model,
        forward_step_func=forward_step_func,
        backward_step_func=backward_step_func,
        data_iterator=data_iter,
        num_microbatches=M,
        pp_size=PP,
        pp_rank=rank,
        pair_groups=pair_groups,
        tensor_shape=(SEQ, 1, HIDDEN),
        dtype=DTYPE,
        device=device,
        infer_model=None,
        infer_local=True,
        loss_fn_for_iF=None,  # skip iF in this smoke
    )
    log_probs, metrics = executor.run(schedule_ops)
    elapsed = time.perf_counter() - t_start
    print(f"[r{rank}] schedule complete @ {elapsed:.1f}s; metrics={metrics}", flush=True)
    print(f"[r{rank}] grad norm: {model.linear.weight.grad.norm().item() if model.linear.weight.grad is not None else 'None'}", flush=True)

    optimizer.step()
    dist.barrier()
    print(f"[r{rank}] all done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
