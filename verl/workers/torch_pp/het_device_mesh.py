"""Lightweight DeviceMesh stand-ins for heterogeneous TP groups.

init_device_mesh is collective (all ranks must use the same shape),
which is impossible with heterogeneous TP (e.g. [[0,1],[2],[3]]).
These wrappers provide the same interface backed by dist.new_group().
"""

from __future__ import annotations

import torch
import torch.distributed as dist


class FakeDeviceMeshDim:
    """Stand-in for device_mesh["infer_tp"] / device_mesh["tp"] etc."""

    def __init__(
        self,
        ranks: list[int],
        local_rank: int,
        group: dist.ProcessGroup | None = None,
    ):
        self.mesh = torch.tensor(ranks, dtype=torch.int64)
        self._local_rank = local_rank
        self._group = group

    def get_local_rank(self) -> int:
        return self._local_rank

    def size(self) -> int:
        return len(self.mesh)

    def get_group(self) -> dist.ProcessGroup | None:
        return self._group


class FakeDeviceMesh:
    """Stand-in for a full DeviceMesh with named dimensions.

    Supports subscript access (mesh["infer_tp"]), get_rank(), size(dim),
    and get_group(name) — everything SGLangRollout and sgl_update_weights need.
    """

    def __init__(
        self,
        dim_map: dict[str, FakeDeviceMeshDim],
        rank: int,
    ):
        self._dim_map = dim_map
        self._rank = rank

    def __getitem__(self, key: str) -> FakeDeviceMeshDim:
        if key in self._dim_map:
            return self._dim_map[key]
        raise KeyError(f"Unknown mesh dimension: {key}")

    def get_rank(self) -> int:
        return self._rank

    def size(self, dim: int) -> int:
        # dim 0 = dp, dim 1 = tp, dim 2 = pp
        keys_by_dim = {0: ("dp",), 1: ("tp", "infer_tp"), 2: ("pp", "infer_pp")}
        for k in keys_by_dim.get(dim, ()):
            if k in self._dim_map:
                return self._dim_map[k].size()
        return 1

    def get_group(self, name: str) -> dist.ProcessGroup | None:
        if name in self._dim_map:
            return self._dim_map[name].get_group()
        raise KeyError(f"Unknown group: {name}")
