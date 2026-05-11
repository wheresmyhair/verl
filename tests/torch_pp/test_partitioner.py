"""Tests for layer partitioning logic."""

import pytest
from verl.workers.torch_pp.partitioner import compute_layer_assignment


class TestComputeLayerAssignment:
    def test_even_split(self):
        result = compute_layer_assignment(28, 4)
        assert result == [(0, 7), (7, 14), (14, 21), (21, 28)]

    def test_uneven_split_extras_go_to_earlier_stages(self):
        # 30 layers / 4 stages = 7 base + 2 extra → stages 0,1 get 8
        result = compute_layer_assignment(30, 4)
        assert result == [(0, 8), (8, 16), (16, 23), (23, 30)]
        # Verify total coverage
        total = sum(end - start for start, end in result)
        assert total == 30

    def test_single_stage(self):
        result = compute_layer_assignment(28, 1)
        assert result == [(0, 28)]

    def test_stages_equal_layers(self):
        result = compute_layer_assignment(4, 4)
        assert result == [(0, 1), (1, 2), (2, 3), (3, 4)]

    def test_no_gaps_or_overlaps(self):
        for num_layers in [12, 24, 28, 32, 48]:
            for pp_size in [1, 2, 4, 8]:
                if pp_size > num_layers:
                    continue
                result = compute_layer_assignment(num_layers, pp_size)
                assert len(result) == pp_size
                # Check contiguous
                for i in range(len(result) - 1):
                    assert result[i][1] == result[i + 1][0]
                # Check full coverage
                assert result[0][0] == 0
                assert result[-1][1] == num_layers
