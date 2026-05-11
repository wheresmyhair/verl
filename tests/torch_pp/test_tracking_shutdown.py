"""Pytest coverage for tracking shutdown behavior."""

from verl.utils.tracking import Tracking


class _FakeBackend:
    def __init__(self, raise_on_finish: bool = False, raise_on_teardown: bool = False):
        self.finish_calls = 0
        self.teardown_calls = 0
        self.raise_on_finish = raise_on_finish
        self.raise_on_teardown = raise_on_teardown

    def finish(self, *args, **kwargs):
        self.finish_calls += 1
        if self.raise_on_finish:
            raise RuntimeError("finish failed")

    def teardown(self, *args, **kwargs):
        self.teardown_calls += 1
        if self.raise_on_teardown:
            raise RuntimeError("teardown failed")


class _FakeSimpleBackend:
    def __init__(self):
        self.finish_calls = 0

    def finish(self, *args, **kwargs):
        self.finish_calls += 1


def _make_tracking(logger_dict):
    tracking = Tracking.__new__(Tracking)
    tracking.logger = logger_dict
    tracking._closed = False
    return tracking


def test_close_calls_finish_and_teardown_for_wandb():
    wandb_backend = _FakeBackend()
    tensorboard_backend = _FakeSimpleBackend()
    tracking = _make_tracking(
        {
            "wandb": wandb_backend,
            "tensorboard": tensorboard_backend,
        }
    )

    tracking.close()

    assert wandb_backend.finish_calls == 1
    assert wandb_backend.teardown_calls == 1
    assert tensorboard_backend.finish_calls == 1
    assert tracking._closed is True


def test_close_is_idempotent():
    wandb_backend = _FakeBackend()
    tracking = _make_tracking({"wandb": wandb_backend})

    tracking.close()
    tracking.close()

    assert wandb_backend.finish_calls == 1
    assert wandb_backend.teardown_calls == 1


def test_close_swallows_finish_and_teardown_errors():
    wandb_backend = _FakeBackend(raise_on_finish=True, raise_on_teardown=True)
    tracking = _make_tracking({"wandb": wandb_backend})

    tracking.close()

    assert wandb_backend.finish_calls == 1
    assert wandb_backend.teardown_calls == 1
    assert tracking._closed is True
