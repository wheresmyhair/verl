"""Pytest coverage for torch-PP worker checkpoint helpers."""

import torch

from verl.workers.torch_pp_workers import ActorRolloutRefWorker


class _FakeStage:
    def __init__(self, state):
        self.state = state
        self.loaded_state = None

    def get_global_state_dict(self):
        return self.state

    def load_state_dict_from_full(self, state):
        self.loaded_state = state


class _FakeOptimizer:
    def __init__(self, state):
        self._state = state
        self.loaded_state = None

    def state_dict(self):
        return self._state

    def load_state_dict(self, state):
        self.loaded_state = state


def _make_worker(pp_rank, stage_state, optimizer_state, local_path=None):
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.train_stage = _FakeStage(stage_state)
    worker.ref_stage = _FakeStage({})
    worker.optimizer = _FakeOptimizer(optimizer_state)
    worker.pp_rank = pp_rank
    worker.pp_size = 2
    worker.local_path = local_path
    worker._is_offload_param = False
    worker._ref_is_offload_param = False
    worker._is_offload_optimizer = False
    worker.device = torch.device("cpu")
    return worker


def test_checkpoint_save_and_load_round_trip(tmp_path, monkeypatch):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "config.json").write_text('{"model":"qwen"}')

    saved_worker = _make_worker(
        pp_rank=0,
        stage_state={"layer0.weight": torch.tensor([1.0, 2.0])},
        optimizer_state={"state": {}, "param_groups": [{"lr": 0.1}]},
        local_path=str(src_dir),
    )

    def fake_all_gather_object(gathered, local_state):
        gathered[:] = [
            local_state,
            {"layer1.weight": torch.tensor([3.0, 4.0])},
        ]

    monkeypatch.setattr("verl.workers.torch_pp_workers.dist.all_gather_object", fake_all_gather_object)
    monkeypatch.setattr("verl.workers.torch_pp_workers.dist.barrier", lambda: None)

    ckpt_dir = tmp_path / "ckpt"
    saved_worker.save_checkpoint(str(ckpt_dir))

    model_path = ckpt_dir / "model.safetensors"
    if not model_path.exists():
        model_path = ckpt_dir / "model.pt"
    assert model_path.exists()
    assert (ckpt_dir / "optimizer_rank0.pt").exists()
    assert (ckpt_dir / "config.json").exists()

    loaded_worker = _make_worker(
        pp_rank=0,
        stage_state={},
        optimizer_state={"state": {}, "param_groups": []},
    )
    monkeypatch.setattr("verl.workers.torch_pp_workers.dist.barrier", lambda: None)

    loaded_worker.load_checkpoint(str(ckpt_dir))

    expected_keys = {"layer0.weight", "layer1.weight"}
    assert set(loaded_worker.train_stage.loaded_state.keys()) == expected_keys
    assert set(loaded_worker.ref_stage.loaded_state.keys()) == expected_keys
    assert loaded_worker.optimizer.loaded_state["param_groups"][0]["lr"] == 0.1
