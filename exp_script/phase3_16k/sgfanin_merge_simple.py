#!/usr/bin/env python3
"""
Simple SGLang fan-in trace merger — just combines per-rank events.

For each step:
  1. Concatenate verl-side step{N}_rank{R}.json per-rank files
     (load_rollout / rollout / compute_log_prob / update_actor boxes recorded
     directly by verl's PPTracer in fsdp_workers.py).
  2. Concatenate sched_rank{R}.jsonl events (window/switch events written
     directly by the sglang scheduler subprocess in set_dynamic_topology
     using real time.time() measurements). Events within this step's
     wall-clock window get their ts converted to step-relative us via
     subtraction from the rank's wall_t0_unix meta event.
  3. Write step{N}_merged.json.

No regex log parsing, no event broadcasting, no synthesized bars. Every
event is real measurement data written directly from one process's
perspective.
"""
import json
import sys
from pathlib import Path


def load_per_rank_verl(trace_dir: Path, step: int, pp_size: int):
    """Return dict {rank: (wall_t0_unix, events)}."""
    out = {}
    for rank in range(pp_size):
        path = trace_dir / f"step{step}_rank{rank}.json"
        if not path.exists():
            continue
        events = json.load(path.open())
        wall_t0 = None
        for e in events:
            if e.get("cat") == "meta" and e.get("args", {}).get("wall_t0_unix") is not None:
                wall_t0 = float(e["args"]["wall_t0_unix"])
                break
        if wall_t0 is not None:
            out[rank] = (wall_t0, events)
    return out


def load_sched_events_by_rank(trace_dir: Path, pp_size: int):
    """Return dict {rank: [event dicts]} read from sched_rank{R}.jsonl."""
    out = {}
    for rank in range(pp_size):
        path = trace_dir / f"sched_rank{rank}.jsonl"
        if not path.exists():
            continue
        events = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        out[rank] = events
    return out


_SCHED_COLORS = {
    ("window", "dp"): "cq_build_passed",  # green — DP topology active
    ("window", "tp"): "rail_response",    # blue — TP topology active
    ("switch", "dp_to_tp"): "terrible",   # red — fan-in switch
    ("switch", "tp_to_dp"): "rail_animation",  # orange — fan-out switch
    ("busy",): "good",                    # dark green — this rank actively decoding
    ("idle",): "rail_idle",               # gray — this rank has no work (bubble)
}


def sched_to_chrome_events(sched_events, wall_t0, rollout_begin_wall, rollout_end_wall, rank):
    """Convert fork scheduler's time.time()-stamped events to Chrome trace
    X events, clipped to the verl `rollout` phase box. Events outside the
    rollout phase are dropped — they belong to other steps or to scheduler
    boot / teardown, not this rank's rollout work."""
    out = []
    for e in sched_events:
        wb = float(e["wall_begin"])
        we = float(e["wall_end"])
        # Drop events fully outside the rollout phase window.
        if we <= rollout_begin_wall or wb >= rollout_end_wall:
            continue
        # Clip to rollout phase boundaries.
        wb_clip = max(wb, rollout_begin_wall)
        we_clip = min(we, rollout_end_wall)
        ts_us = (wb_clip - wall_t0) * 1e6
        dur_us = max(0.0, (we_clip - wb_clip) * 1e6)
        kind = e["kind"]
        if kind == "window":
            # Window events are topology-level time ranges. Now that
            # busy/idle events provide finer-grained work/idle bars
            # labeled with the active topology, we skip window events
            # to avoid overlapping bars on the Phases row.
            continue
        elif kind == "switch":
            frm = e["from"]
            to = e["to"]
            direction = f"{frm}_to_{to}"
            name = f"{direction}"
            cat = f"{direction}"  # no slash
            cname = _SCHED_COLORS[(kind, direction)]
            # Switches are <15ms — invisible as X-duration events at
            # full-step zoom and create overlaps with adjacent bars.
            # Emit as instant markers (vertical lines, always visible).
            out.append({
                "name": name,
                "cat": cat,
                "ph": "i",
                "s": "t",
                "ts": ts_us,
                "pid": f"0 Phases GPU {rank}",
                "tid": "phases",
                "args": {
                    "pp_rank": rank,
                    "wall_begin_unix": wb,
                    "wall_end_unix": we,
                    "dur_s": we - wb,
                    "dur_ms": (we - wb) * 1000,
                    **{k: v for k, v in e.items() if k not in ("kind",)},
                },
                "cname": cname,
            })
            continue
        elif kind in ("busy", "idle"):
            # Determine which topology was active during this busy/idle
            # period by checking if its midpoint falls within a dp or tp
            # window event from the same rank's sched data.
            mid_wall = (wb + we) / 2
            topo = "dp"  # default
            for w in sched_events:
                if w["kind"] == "window" and w["rank"] == rank:
                    if w["wall_begin"] <= mid_wall <= w["wall_end"]:
                        topo = w["topo"]
                        break
            if kind == "busy":
                name = f"{topo}_decode"
                cat = f"{topo}_decode"  # no slash — Perfetto may filter "a/b" cats
                cname = _SCHED_COLORS[("window", topo)]
            else:
                name = "idle (bubble)"
                cat = "idle"
                cname = _SCHED_COLORS[(kind,)]
            # Put on Phases row (same as load_rollout / tp_decode) so
            # the user sees work + idle in one timeline per rank.
            pid = f"0 Phases GPU {rank}"
            tid = "phases"
        else:
            continue
        out.append({
            "name": name,
            "cat": cat,
            "ph": "X",
            "ts": int(ts_us),
            "dur": int(dur_us),
            "pid": pid,
            "tid": tid,
            "args": {"dur_s": round(we - wb, 3), "rank": rank},
            "cname": cname,
        })
    return out


def _normalize_tracks(events: list, pp_size: int) -> list:
    """Group per-rank pid labels into stable Perfetto processes with ordered threads."""
    normalized = []
    for event in events:
        event = dict(event)
        pid = event.get("pid")
        if not isinstance(pid, str):
            normalized.append(event)
            continue
        if pid.startswith("0 Phases GPU "):
            rank = int(pid.rsplit("GPU ", 1)[1])
            event["pid"] = "0 Phases"
            event["tid"] = f"GPU {rank}"
        elif pid.startswith("1 GPU ") and "GPU Ops" not in pid:
            rank = int(pid.rsplit("GPU ", 1)[1])
            event["pid"] = "1 GPU Ops"
            event["tid"] = f"GPU {rank}"
        elif pid.startswith("5 HBM GPU "):
            rank = int(pid.rsplit("GPU ", 1)[1])
            event["pid"] = "5 HBM"
            event["tid"] = f"GPU {rank}"
        normalized.append(event)
    return normalized


def _process_meta(pp_size: int) -> list:
    meta = []
    process_sort = {"0 Phases": 0, "1 GPU Ops": 1, "5 HBM": 2}
    for pid, sort_index in process_sort.items():
        meta.append({"ph": "M", "name": "process_name", "pid": pid, "tid": 0,
                     "args": {"name": pid}})
        meta.append({"ph": "M", "name": "process_sort_index", "pid": pid, "tid": 0,
                     "args": {"sort_index": sort_index}})
        for r in range(pp_size):
            tid = f"GPU {r}"
            meta.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid,
                         "args": {"name": tid}})
            meta.append({"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": tid,
                         "args": {"sort_index": r}})
    return meta


def merge_step(trace_dir: Path, step: int, pp_size: int, sched_by_rank):
    verl = load_per_rank_verl(trace_dir, step, pp_size)
    if not verl:
        return None

    all_events = []
    for rank, (wall_t0, rank_events) in verl.items():
        # Drop any previously-added sched events (in case we're re-merging)
        filtered = [
            e for e in rank_events
            if not e.get("cat", "").startswith("rollout/")
        ]
        all_events.extend(filtered)
        # Find the verl-side rollout phase box for THIS rank — sched events
        # are clipped to this window so they don't leak into compute_log_prob
        # / update_actor / the next step's rollout.
        rollout_box = next(
            (e for e in filtered
             if e.get("ph") == "X" and e.get("cat") == "rollout"),
            None,
        )
        if rollout_box is None:
            # No rollout on this rank (e.g. ref-only worker), skip sched.
            continue
        rollout_begin_wall = wall_t0 + rollout_box["ts"] / 1e6
        rollout_end_wall = wall_t0 + (rollout_box["ts"] + rollout_box["dur"]) / 1e6
        rank_sched = sched_by_rank.get(rank, [])
        chrome_sched = sched_to_chrome_events(
            rank_sched, wall_t0, rollout_begin_wall, rollout_end_wall, rank,
        )
        if chrome_sched:
            # Sub-phases from fork scheduler exist — remove the opaque verl
            # rollout box and replace it with the sub-phase breakdown.
            all_events = [e for e in all_events if e is not rollout_box]
            all_events.extend(chrome_sched)
        # else (baseline / no fan-in): keep the rollout box as-is.

    # Normalize pid/tid for stable track layout
    all_events = _normalize_tracks(all_events, pp_size)
    all_events = _process_meta(pp_size) + all_events

    out_path = trace_dir / f"step{step}_merged.json"
    out_path.write_text(json.dumps(all_events))
    return out_path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    trace_dir = Path(sys.argv[1])
    pp_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    # Scan for step numbers
    steps = set()
    for f in trace_dir.glob("step*_rank*.json"):
        try:
            step = int(f.stem.split("_")[0].replace("step", ""))
            steps.add(step)
        except Exception:
            pass
    sched_by_rank = load_sched_events_by_rank(trace_dir, pp_size)
    print(f"loaded sched events: {[f'rank{r}={len(v)}' for r, v in sorted(sched_by_rank.items())]}")

    for step in sorted(steps):
        out = merge_step(trace_dir, step, pp_size, sched_by_rank)
        if out is not None:
            print(f"  step {step}: {out}")


if __name__ == "__main__":
    main()
