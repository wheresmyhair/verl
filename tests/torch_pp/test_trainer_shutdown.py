"""Pytest coverage for trainer cleanup helpers."""

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class _FakeIterator:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def _shutdown_workers(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")


class _FakeLoader:
    def __init__(self, iterator=None):
        self._iterator = iterator


def test_shutdown_dataloader_workers_clears_iterators_even_on_error(capsys):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    train_iter = _FakeIterator()
    val_iter = _FakeIterator(fail=True)
    trainer.train_dataloader = _FakeLoader(train_iter)
    trainer.val_dataloader = _FakeLoader(val_iter)

    trainer._shutdown_dataloader_workers()

    assert train_iter.calls == 1
    assert val_iter.calls == 1
    assert trainer.train_dataloader._iterator is None
    assert trainer.val_dataloader._iterator is None
    assert "failed to shutdown val_dataloader workers cleanly" in capsys.readouterr().out
