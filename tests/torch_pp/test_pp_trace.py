"""Pytest coverage for PP trace helpers."""

import json

from verl.workers.torch_pp.pp_trace import PPTracer


def test_begin_step_applies_time_offset():
    tracer = PPTracer(pp_rank=0, pp_size=4, enabled=True)
    tracer.begin_step(step=3, time_offset_us=1234.0)

    with tracer.phase("compute_ref_log_prob"):
        pass

    event = tracer.events[-1]
    assert event["ts"] >= 1234.0
    assert event["args"]["step"] == 3


def test_save_appends_to_existing_trace_file(tmp_path):
    tracer_a = PPTracer(pp_rank=0, pp_size=4, enabled=True, save_dir=str(tmp_path))
    tracer_a.begin_step(step=1)
    with tracer_a.phase("compute_log_prob"):
        pass
    tracer_a.end_step()

    tracer_b = PPTracer(pp_rank=0, pp_size=4, enabled=True, save_dir=str(tmp_path))
    tracer_b.begin_step(step=1, time_offset_us=1000.0)
    with tracer_b.phase("compute_ref_log_prob"):
        pass
    tracer_b.end_step()

    path = tmp_path / "step1_rank0.json"
    events = json.loads(path.read_text())
    cats = [event["cat"] for event in events if event.get("ph") == "X"]

    assert "compute_log_prob" in cats
    assert "compute_ref_log_prob" in cats
    assert cats.count("compute_log_prob") == 1
    assert cats.count("compute_ref_log_prob") == 1
