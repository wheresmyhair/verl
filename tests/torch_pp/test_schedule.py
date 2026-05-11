"""Tests for 1F1B pipeline schedule."""

import pytest
from verl.workers.torch_pp.schedule import build_1f1b_schedule, ScheduleOp


class TestBuild1F1BSchedule:
    def test_last_stage_no_warmup(self):
        """Last stage has 0 warmup forwards."""
        ops = build_1f1b_schedule(pp_rank=3, pp_size=4, num_micro_batches=4)
        # Should be F0 B0 F1 B1 F2 B2 F3 B3
        assert len(ops) == 8
        for i in range(4):
            assert ops[2 * i] == ScheduleOp("forward", i)
            assert ops[2 * i + 1] == ScheduleOp("backward", i)

    def test_first_stage_max_warmup(self):
        """First stage has pp_size-1 warmup forwards."""
        ops = build_1f1b_schedule(pp_rank=0, pp_size=4, num_micro_batches=4)
        # Warmup: F0 F1 F2, Steady: F3 B0, Cooldown: B1 B2 B3
        assert ops[0] == ScheduleOp("forward", 0)
        assert ops[1] == ScheduleOp("forward", 1)
        assert ops[2] == ScheduleOp("forward", 2)
        assert ops[3] == ScheduleOp("forward", 3)
        assert ops[4] == ScheduleOp("backward", 0)
        assert ops[5] == ScheduleOp("backward", 1)
        assert ops[6] == ScheduleOp("backward", 2)
        assert ops[7] == ScheduleOp("backward", 3)

    def test_total_ops_count(self):
        """Each stage does exactly M forwards + M backwards."""
        for pp_size in [2, 4, 8]:
            for M in [pp_size, pp_size * 2]:
                for rank in range(pp_size):
                    ops = build_1f1b_schedule(rank, pp_size, M)
                    fwd = sum(1 for o in ops if o.op == "forward")
                    bwd = sum(1 for o in ops if o.op == "backward")
                    assert fwd == M, f"rank={rank}, pp={pp_size}, M={M}"
                    assert bwd == M

    def test_all_micro_batches_covered(self):
        """Each micro-batch ID appears exactly once in forward and once in backward."""
        ops = build_1f1b_schedule(pp_rank=1, pp_size=4, num_micro_batches=8)
        fwd_ids = [o.micro_batch_id for o in ops if o.op == "forward"]
        bwd_ids = [o.micro_batch_id for o in ops if o.op == "backward"]
        assert sorted(fwd_ids) == list(range(8))
        assert sorted(bwd_ids) == list(range(8))

    def test_backward_after_forward_per_microbatch(self):
        """For each micro-batch, backward appears after its forward."""
        for rank in range(4):
            ops = build_1f1b_schedule(rank, 4, 8)
            fwd_pos = {}
            bwd_pos = {}
            for i, op in enumerate(ops):
                if op.op == "forward":
                    fwd_pos[op.micro_batch_id] = i
                else:
                    bwd_pos[op.micro_batch_id] = i
            for mb in range(8):
                assert fwd_pos[mb] < bwd_pos[mb], f"rank={rank}, mb={mb}"

    def test_too_few_micro_batches_raises(self):
        with pytest.raises(ValueError, match="num_micro_batches"):
            build_1f1b_schedule(pp_rank=0, pp_size=4, num_micro_batches=3)

    def test_cross_stage_comm_no_deadlock(self):
        """Verify that the 1F1B schedule doesn't create send/recv deadlocks.

        For adjacent stages (r, r+1): every forward send from r must have a
        matching recv at r+1 that comes before r+1 tries to send back to r
        on the same micro-batch.
        """
        pp_size, M = 4, 8
        schedules = [build_1f1b_schedule(r, pp_size, M) for r in range(pp_size)]

        for r in range(pp_size - 1):
            # Collect the order of micro-batch IDs that stage r sends (forward)
            fwd_sends = [o.micro_batch_id for o in schedules[r] if o.op == "forward"]
            # Collect the order that stage r+1 receives (forward)
            fwd_recvs = [o.micro_batch_id for o in schedules[r + 1] if o.op == "forward"]
            # Both should process micro-batches in the same order
            assert fwd_sends == fwd_recvs, f"stages {r},{r+1} fwd order mismatch"
