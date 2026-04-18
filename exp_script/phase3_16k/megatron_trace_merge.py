#!/usr/bin/env python3
"""Merge per-rank Megatron PP trace JSONs into one Perfetto-loadable file.

Layout matches ~/perfetto_example.json:
  - pid "0 Phases"  — tid GPU 0..3: phase-level bars (rollout, ref, fused_update_actor, ...)
  - pid "1 GPU Ops" — tid GPU 0..3: per-mb ops (iF, tF, tB, p2p, optimizer)
  - pid "5 HBM"     — tid GPU 0..3: memory counter events

Fixes the timestamp bug where fused_update_actor events use a different
time origin (PPTracer _step_start reset by ref's end_step). Detects the
break and re-aligns by anchoring fused_update_actor's start to the end
of the last preceding phase.

Usage:
    python megatron_trace_merge.py /path/to/profiling_dir [--step 1]
"""
import argparse
import json
import os


def merge(profiling_dir: str, step: int = 1):
    merged = []

    # Process metadata — one process, all ranks as threads
    for pid, pname in [("0 Phases", "0 Phases"), ("1 GPU Ops", "1 GPU Ops"), ("5 HBM", "5 HBM")]:
        merged.append({"name": "process_name", "ph": "M", "pid": pid, "args": {"name": pname}})
        merged.append({"name": "process_sort_index", "ph": "M", "pid": pid, "args": {"sort_index": int(pid.split()[0])}})
        for rank in range(4):
            tid = f"GPU {rank}"
            merged.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": tid, "args": {"name": tid}})
            merged.append({"name": "thread_sort_index", "ph": "M", "pid": pid, "tid": tid, "args": {"sort_index": rank}})

    for rank in range(4):
        fname = os.path.join(profiling_dir, f"step{step}_rank{rank}.json")
        if not os.path.exists(fname):
            print(f"[warn] {fname} not found, skipping rank {rank}")
            continue

        with open(fname) as f:
            data = json.load(f)
        events = data if isinstance(data, list) else data.get("traceEvents", [])

        tid = f"GPU {rank}"

        # Separate events into groups
        phase_events = []   # tid=phases
        ops_events = []     # tid=ops
        hbm_events = []     # ph=C (counter)
        for e in events:
            if e.get("ph") == "M":
                continue  # skip old metadata
            if e.get("ph") == "C":
                hbm_events.append(e)
            elif e.get("tid") == "phases":
                phase_events.append(e)
            elif e.get("tid") == "ops":
                ops_events.append(e)

        # ----------------------------------------------------------
        # Fix timestamp misalignment: detect if fused_update_actor
        # events have a different time origin than rollout/ref.
        #
        # Strategy: find the end of the last "pre-fused" phase (e.g.
        # compute_ref_log_prob or unload_ref) and the start of
        # fused_update_actor. If fused_update_actor starts near 0
        # while pre-fused phases end at ~80s, compute the offset.
        # ----------------------------------------------------------
        pre_fused_names = {
            "load_rollout", "rollout", "unload_rollout",
            "load_ref", "compute_ref_log_prob_inner", "unload_ref",
            "compute_ref_log_prob",
        }
        fused_names = {
            "fused_update_actor", "load_training", "sync_fused_replica",
            "infer_forward", "update_policy", "unload_training",
        }

        pre_fused_end = 0.0
        fused_start = None
        for e in phase_events:
            name = e.get("name", "")
            ts = e.get("ts", 0)
            dur = e.get("dur", 0)
            if name in pre_fused_names:
                pre_fused_end = max(pre_fused_end, ts + dur)
            if name == "fused_update_actor" and fused_start is None:
                fused_start = ts

        # If fused group starts near 0 while pre-fused ends >> 0, apply offset
        offset = 0.0
        if fused_start is not None and pre_fused_end > 1e6 and fused_start < 1e6:
            # fused_update_actor is in a different time coordinate
            # Add a small gap (0.5s) after the last pre-fused phase
            offset = pre_fused_end + 500_000  # 0.5s gap in us
            if rank == 0:
                print(f"[fix] rank {rank}: fused_start={fused_start/1e6:.2f}s, "
                      f"pre_fused_end={pre_fused_end/1e6:.2f}s, offset={offset/1e6:.2f}s")

        def needs_offset(e):
            name = e.get("name", "").split(" mb=")[0]
            cat = e.get("cat", "")
            # All events that belong to the fused_update_actor group
            if name in fused_names or cat in fused_names:
                return True
            # GPU ops (tF, tB, iF, p2p, optimizer) that are inside update_policy
            if cat in ("train_forward", "train_backward", "infer_forward",
                       "p2p_send", "p2p_recv", "p2p_send_recv", "optimizer"):
                # Only offset if their timestamp is in the "wrong" coordinate
                ts = e.get("ts", 0)
                if ts < 1e6 and pre_fused_end > 1e6:
                    return True
            return False

        # Emit phase events
        for e in phase_events:
            out = {"ph": "X", "pid": "0 Phases", "tid": tid}
            out["name"] = e.get("name", "")
            out["cat"] = e.get("cat", "")
            ts = e.get("ts", 0)
            dur = e.get("dur", 0)
            if offset > 0 and needs_offset(e):
                ts += offset
            out["ts"] = int(ts)
            out["dur"] = int(dur)
            if "args" in e:
                out["args"] = e["args"]
            if "cname" in e:
                out["cname"] = e["cname"]
            merged.append(out)

        # Emit ops events
        for e in ops_events:
            out = {"ph": "X", "pid": "1 GPU Ops", "tid": tid}
            out["name"] = e.get("name", "")
            out["cat"] = e.get("cat", "")
            ts = e.get("ts", 0)
            dur = e.get("dur", 0)
            if offset > 0 and needs_offset(e):
                ts += offset
            out["ts"] = int(ts)
            out["dur"] = int(dur)
            if "args" in e:
                out["args"] = e["args"]
            if "cname" in e:
                out["cname"] = e["cname"]
            merged.append(out)

        # Emit HBM events
        for e in hbm_events:
            out = dict(e)
            out["pid"] = "5 HBM"
            out["tid"] = tid
            ts = out.get("ts", 0)
            if offset > 0 and ts < 1e6 and pre_fused_end > 1e6:
                ts += offset
            out["ts"] = int(ts)
            merged.append(out)

    out_path = os.path.join(profiling_dir, f"step{step}_merged.json")
    with open(out_path, "w") as f:
        json.dump(merged, f)
    print(f"Wrote {out_path} ({len(merged)} events)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("profiling_dir")
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()
    merge(args.profiling_dir, args.step)
