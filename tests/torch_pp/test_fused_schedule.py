"""Tests for the fused forward schedule generator and validator."""

import pytest

from verl.workers.torch_pp.fused_schedule import (
    FusedScheduleOp,
    build_default_fused_schedule,
    parse_schedule,
    print_fused_schedule,
    validate_schedule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_generated(pp_size: int, num_micro_batches: int):
    """Generate a schedule and run full validation."""
    s = build_default_fused_schedule(pp_size, num_micro_batches)
    parsed = {r: parse_schedule(ops) for r, ops in s.items()}
    validate_schedule(parsed, pp_size, num_micro_batches)
    return s


# ---------------------------------------------------------------------------
# Parse / repr round-trip
# ---------------------------------------------------------------------------


class TestParse:
    def test_parse_roundtrip(self):
        tokens = ["tF.0", "tB.3", "iF.7"]
        ops = parse_schedule(tokens)
        assert [repr(o) for o in ops] == tokens

    def test_parse_invalid_prefix(self):
        with pytest.raises(ValueError, match="Invalid schedule token"):
            parse_schedule(["xF.0"])

    def test_parse_invalid_mb(self):
        with pytest.raises(ValueError, match="Invalid micro-batch"):
            parse_schedule(["tF.abc"])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_tex_schedule_validates(self):
        """The hand-written tex example (P=4, M=4) must pass validation."""
        tex = {
            0: "tF.0 tF.1 tF.2 iF.0 tF.3 iF.1 iF.2 iF.3 tB.0 tB.1 tB.2 tB.3".split(),
            1: "tF.0 iF.0 tF.1 iF.1 tF.2 iF.2 tF.3 iF.3 tB.0 tB.1 tB.2 tB.3".split(),
            2: "iF.0 tF.0 iF.1 tF.1 iF.2 tF.2 iF.3 tB.0 tB.1 tF.3 tB.2 tB.3".split(),
            3: "iF.0 iF.1 iF.2 iF.3 tF.0 tB.0 tF.1 tB.1 tF.2 tB.2 tF.3 tB.3".split(),
        }
        parsed = {r: parse_schedule(ops) for r, ops in tex.items()}
        validate_schedule(parsed, 4, 4)

    def test_wrong_op_count_rejected(self):
        bad = {
            0: parse_schedule(["tF.0", "tB.0", "iF.0"]),
            1: parse_schedule(["tF.0", "tB.0"]),  # missing iF.0
        }
        with pytest.raises(ValueError, match="expected 1"):
            validate_schedule(bad, 2, 1)

    def test_missing_dependency_rejected(self):
        """tB before tF on the same rank → dependency violation."""
        bad = {
            0: parse_schedule(["iF.0", "tB.0", "tF.0"]),
        }
        with pytest.raises(ValueError, match="dependency"):
            validate_schedule(bad, 1, 1)


# ---------------------------------------------------------------------------
# Default schedule generator — structural properties
# ---------------------------------------------------------------------------


class TestDefaultSchedule:
    @pytest.mark.parametrize(
        "pp_size,num_micro_batches",
        [(1, 4), (2, 4), (3, 6), (4, 4), (4, 8), (8, 4)],
    )
    def test_validates(self, pp_size, num_micro_batches):
        _validate_generated(pp_size, num_micro_batches)

    @pytest.mark.parametrize("pp_size", [1, 2, 4, 8])
    def test_last_rank_iF_first(self, pp_size):
        """Last rank (first inference stage) runs all iF before any tF/tB."""
        M = 4
        s = build_default_fused_schedule(pp_size, M)
        last = s[pp_size - 1]
        ops = parse_schedule(last)
        # First M ops should all be infer_forward
        for i in range(M):
            assert ops[i].op == "infer_forward", (
                f"Last rank op {i} should be iF, got {ops[i]}"
            )
            assert ops[i].micro_batch_id == i

    @pytest.mark.parametrize("pp_size", [1, 2, 4])
    def test_last_rank_1f1b_after_iF(self, pp_size):
        """After all iF, last rank runs 1F1B: tF.0 tB.0 tF.1 tB.1 ..."""
        M = 4
        s = build_default_fused_schedule(pp_size, M)
        last = s[pp_size - 1]
        ops = parse_schedule(last)
        train_ops = ops[M:]  # skip iF phase
        for i in range(M):
            assert train_ops[2 * i].op == "train_forward"
            assert train_ops[2 * i].micro_batch_id == i
            assert train_ops[2 * i + 1].op == "train_backward"
            assert train_ops[2 * i + 1].micro_batch_id == i

    def test_op_counts(self):
        """Each rank should have exactly M of each op type."""
        for P, M in [(2, 4), (4, 8)]:
            s = build_default_fused_schedule(P, M)
            for r in range(P):
                ops = parse_schedule(s[r])
                counts = {"train_forward": 0, "train_backward": 0, "infer_forward": 0}
                for op in ops:
                    counts[op.op] += 1
                for op_type, count in counts.items():
                    assert count == M, f"Rank {r}: {op_type} count {count} != {M}"

    def test_priority_iF_over_tB_when_tied(self):
        """When iF and tB are ready at the same mb index, iF should come first."""
        # With P=4, rank 2 should have many interleaved iF/tF pairs.
        # Verify: whenever iF.m and tB.m both appear, iF.m comes first.
        M = 4
        s = build_default_fused_schedule(4, M)
        for r in range(4):
            ops = parse_schedule(s[r])
            first_seen = {}
            for op in ops:
                key = (op.op, op.micro_batch_id)
                if key not in first_seen:
                    first_seen[key] = len(first_seen)
            # For each mb, iF should appear before tB
            for mb in range(M):
                iF_pos = first_seen.get(("infer_forward", mb))
                tB_pos = first_seen.get(("train_backward", mb))
                assert iF_pos is not None and tB_pos is not None
                assert iF_pos < tB_pos, (
                    f"Rank {r}: iF.{mb} at pos {iF_pos} should be before "
                    f"tB.{mb} at pos {tB_pos}"
                )


# ---------------------------------------------------------------------------
# P=1 special case
# ---------------------------------------------------------------------------


class TestP1:
    def test_p1_schedule(self):
        """P=1: all iF first, then tF/tB interleaved (1F1B)."""
        s = build_default_fused_schedule(1, 4)
        assert s[0] == [
            "iF.0", "iF.1", "iF.2", "iF.3",
            "tF.0", "tB.0", "tF.1", "tB.1",
            "tF.2", "tB.2", "tF.3", "tB.3",
        ]


# ---------------------------------------------------------------------------
# Pretty-print (smoke test)
# ---------------------------------------------------------------------------


class TestPrint:
    def test_print_does_not_crash(self, capsys):
        s = build_default_fused_schedule(4, 4)
        print_fused_schedule(s)
        captured = capsys.readouterr()
        assert "Rank" in captured.out
        assert "tF.0" in captured.out
