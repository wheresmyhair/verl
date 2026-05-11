"""Pytest coverage for rollout routing policies used by torch_pp PPO trainer."""

from verl.trainer.ppo.ray_trainer import RolloutRouter


def test_round_robin_routing():
    router = RolloutRouter(num_workers=3, strategy="round_robin")

    assignment, info = router.route(
        sample_indices=[10, 11, 12, 13, 14],
        prompt_lengths=[1, 1, 1, 1, 1],
        epoch=0,
    )

    assert assignment == {0: [0, 3], 1: [1, 4], 2: [2]}
    assert info["strategy"] == "round_robin"


def test_random_routing_is_deterministic_with_seed():
    router_a = RolloutRouter(num_workers=3, strategy="random", random_seed=7)
    router_b = RolloutRouter(num_workers=3, strategy="random", random_seed=7)

    assignment_a, _ = router_a.route(
        sample_indices=[0, 1, 2, 3, 4, 5],
        prompt_lengths=[1, 1, 1, 1, 1, 1],
        epoch=2,
    )
    assignment_b, _ = router_b.route(
        sample_indices=[0, 1, 2, 3, 4, 5],
        prompt_lengths=[1, 1, 1, 1, 1, 1],
        epoch=2,
    )

    assert assignment_a == assignment_b


def test_weighted_response_length_prefers_stronger_group():
    router = RolloutRouter(
        num_workers=3,
        strategy="weighted_response_length",
        worker_weights=[2.0, 1.0, 1.0],
        prompt_coef=0.0,
        response_coef=1.0,
        default_response_length=1.0,
        warmup_epochs=0,
    )
    router.sample_history_estimates = {
        100: 1000.0,
        101: 900.0,
        102: 100.0,
        103: 100.0,
    }

    assignment, info = router.route(
        sample_indices=[100, 101, 102, 103],
        prompt_lengths=[1, 1, 1, 1],
        epoch=1,
    )

    assert 0 in assignment[0]
    assert sorted(assignment[2]) == [2, 3]
    assert info["strategy"] == "weighted_response_length"


def test_warmup_epochs_fallback_to_round_robin():
    router = RolloutRouter(
        num_workers=3,
        strategy="weighted_response_length",
        warmup_epochs=2,
    )
    router.sample_history_estimates = {0: 9999.0}

    assignment, info = router.route(
        sample_indices=[0, 1, 2, 3],
        prompt_lengths=[1, 1, 1, 1],
        epoch=0,
    )

    assert info["strategy"] == "round_robin"
    assert assignment == {0: [0, 3], 1: [1], 2: [2]}


def test_response_aggregation_max():
    router = RolloutRouter(num_workers=2, response_agg="max")
    updated = router.update_history({7: [12.0, 30.0, 18.0]})

    assert updated[7] == 30.0
    assert router.sample_history_estimates[7] == 30.0


def test_response_aggregation_quantile():
    router = RolloutRouter(num_workers=2, response_agg="p80")
    updated = router.update_history({7: [10.0, 20.0, 30.0, 40.0, 50.0]})

    assert updated[7] == 42.0


def test_history_estimator_latest():
    router = RolloutRouter(num_workers=2, history_estimator="latest")
    router.update_history({1: [10.0, 20.0]})
    router.update_history({1: [5.0, 6.0]})

    assert router.sample_history_estimates[1] == 6.0


def test_history_estimator_mean():
    router = RolloutRouter(num_workers=2, history_estimator="mean", response_agg="mean")
    router.update_history({1: [10.0, 20.0]})
    router.update_history({1: [30.0, 50.0]})

    assert router.sample_history_estimates[1] == 27.5


def test_history_estimator_ema():
    router = RolloutRouter(
        num_workers=2,
        history_estimator="ema",
        ema_alpha=0.5,
        response_agg="mean",
    )
    router.update_history({1: [10.0, 20.0]})
    router.update_history({1: [30.0, 50.0]})

    assert router.sample_history_estimates[1] == 27.5


def test_length_strategy_uses_prompt_and_response_coefficients():
    router = RolloutRouter(
        num_workers=2,
        strategy="length",
        worker_weights=[1.0, 1.0],
        prompt_coef=1.0,
        response_coef=2.0,
        warmup_epochs=0,
    )
    router.sample_history_estimates = {
        0: 100.0,
        1: 10.0,
    }

    _, info = router.route(
        sample_indices=[0, 1],
        prompt_lengths=[50.0, 100.0],
        epoch=1,
    )

    assert info["predicted_scores"] == [250.0, 120.0]
