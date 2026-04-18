#!/usr/bin/env python3
"""
Post-run analyzer for SGLang fan-in audit.

Parses verl training logs from 4 experiments:
  1. 1.7B baseline (stock DP)
  2. 1.7B fan-in  (fork dynamic-TP)
  3. 7B baseline
  4. 7B fan-in

Extracts timing_s/*, response_length/*, routing/*, perf/* metrics per step,
plus fan-in topology switch events from sglang scheduler log lines, and
emits:
  - A/B comparison table as Markdown to stdout
  - A chrome-trace JSON per experiment at profiling_sgfanin_audit_*/trace.json
  - A merged 4-process chrome trace at /tmp/sgfanin_audit_merged.json
    (loadable in https://ui.perfetto.dev for visual comparison)
"""
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

VERL_DIR = Path("/home/user/rlpipe/verl")
PROFILING_ROOT = Path("/home/user")

EXPERIMENTS = [
    {
        "name": "baseline-1.7B",
        "label": "Exp 1: Qwen3-1.7B baseline",
        "log": VERL_DIR / "log_p3_sgfanin_audit_baseline_qwen3_1_7b.txt",
        "profiling_dir": PROFILING_ROOT / "profiling_sgfanin_audit_baseline_qwen3_1_7b",
        "pid": 0,
        "fanin": False,
    },
    {
        "name": "fanin-1.7B",
        "label": "Exp 2: Qwen3-1.7B fan-in",
        "log": VERL_DIR / "log_p3_sgfanin_audit_fanin_qwen3_1_7b.txt",
        "profiling_dir": PROFILING_ROOT / "profiling_sgfanin_audit_fanin_qwen3_1_7b",
        "pid": 1,
        "fanin": True,
    },
    {
        "name": "baseline-7B",
        "label": "Exp 3: 7B baseline",
        "log": VERL_DIR / "log_p3_sgfanin_audit_baseline_7b.txt",
        "profiling_dir": PROFILING_ROOT / "profiling_sgfanin_audit_baseline_7b",
        "pid": 2,
        "fanin": False,
    },
    {
        "name": "fanin-7B",
        "label": "Exp 4: 7B fan-in",
        "log": VERL_DIR / "log_p3_sgfanin_audit_fanin_7b.txt",
        "profiling_dir": PROFILING_ROOT / "profiling_sgfanin_audit_fanin_7b",
        "pid": 3,
        "fanin": True,
    },
]

# Metrics to extract from each step's verl metrics line
METRIC_FIELDS = [
    "timing_s/step",
    "timing_s/gen",
    "timing_s/generate_sequences",
    "timing_s/generation_timing/max",
    "timing_s/generation_timing/min",
    "timing_s/generation_timing/topk_ratio",
    "timing_s/old_log_prob",
    "timing_s/update_actor",
    "timing_s/testing",
    "response_length/mean",
    "response_length/max",
    "response_length/min",
    "response_length/clip_ratio",
    "routing/realized_imbalance_ratio",
    "perf/max_memory_allocated_gb",
    "perf/throughput",
    "perf/total_num_tokens",
    "global_seqlen/min",
    "global_seqlen/max",
    "training/global_step",
]

METRIC_RE = re.compile(r"(\S+?):([\-\d.eE+]+)")
STEP_LINE_RE = re.compile(r"step:(\d+)\s*-\s*")
# Match scheduler log lines for topology switches. Example:
# (WorkerDict pid=591012) [2026-04-14 04:45:43 TP1] [rlpipe dynamic-tp] set_dynamic_topology: dp -> tp (strategy=reprefill)
# (WorkerDict pid=591012) [2026-04-14 04:45:50 TP2] [rlpipe dynamic-tp] scheduler rank 2 switched to topology 'tp' (max_total_num_tokens=319885)
TS_SWITCH_BEGIN_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+TP(\d+)\]\s+\[rlpipe dynamic-tp\]\s+"
    r"set_dynamic_topology:\s+(\S+)\s*->\s*(\S+)\s*\(strategy=(\S+)\)"
)
TS_SWITCH_DONE_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+TP(\d+)\]\s+\[rlpipe dynamic-tp\]\s+"
    r"scheduler rank (\d+) switched to topology '(\S+)'\s+\(max_total_num_tokens=(\d+)\)"
)


def parse_log(log_path: Path):
    """Return dict with:
    - steps: list of per-step metrics dicts (ordered by global_step)
    - switch_events: list of (dt, rank, begin|done, from_topo, to_topo, extra)
    - finished: bool, whether we saw step:3 metrics
    - errors: list of str (OOM, RayTaskError, etc)
    """
    out = {"steps": [], "switch_events": [], "finished": False, "errors": []}
    if not log_path.exists():
        out["errors"].append(f"log file missing: {log_path}")
        return out
    text = log_path.read_text(errors="ignore")
    for line in text.splitlines():
        # Training metrics line: ... step:1 - metric1:v1 - metric2:v2 - ...
        m = STEP_LINE_RE.search(line)
        if m and "timing_s/step" in line:
            step = {"_step_num": int(m.group(1))}
            for km, vm in METRIC_RE.findall(line):
                if km in METRIC_FIELDS:
                    try:
                        step[km] = float(vm)
                    except ValueError:
                        pass
            # Also grab routing/realized_response_load/group_[0-3] explicitly
            for g in range(4):
                rm = re.search(r"routing/realized_response_load/group_{}:([\d.]+)".format(g), line)
                if rm:
                    step[f"routing/realized_response_load/group_{g}"] = float(rm.group(1))
            out["steps"].append(step)
        # Topology switch begin
        m = TS_SWITCH_BEGIN_RE.search(line)
        if m:
            ts, rank, frm, to, strategy = m.groups()
            out["switch_events"].append({
                "kind": "begin",
                "ts_str": ts,
                "dt": datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"),
                "rank": int(rank),
                "from": frm,
                "to": to,
                "strategy": strategy,
            })
            continue
        m = TS_SWITCH_DONE_RE.search(line)
        if m:
            ts, tp_label, rank, topo, max_tokens = m.groups()
            out["switch_events"].append({
                "kind": "done",
                "ts_str": ts,
                "dt": datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"),
                "rank": int(rank),
                "topo": topo,
                "max_tokens": int(max_tokens),
            })
            continue
        if "OutOfMemoryError" in line and "Tried to allocate" in line:
            out["errors"].append("CUDA OOM")
        if "RayTaskError" in line and "OutOfMemoryError" in line:
            pass  # already captured
        if "illegal memory access" in line:
            out["errors"].append("CUDA illegal memory access")
    out["errors"] = list(dict.fromkeys(out["errors"]))  # dedup preserving order
    # "finished" means all 3 steps emitted metrics
    if len({s["_step_num"] for s in out["steps"]}) >= 3:
        out["finished"] = True
    return out


def fmt_td(sec):
    if sec is None or (isinstance(sec, float) and sec != sec):
        return "   -"
    return f"{sec:6.1f}"


def print_comparison_table(parsed):
    """Print a Markdown A/B table comparing baseline vs fan-in per step + mean."""
    # Group into scales
    for scale in ("1.7B", "7B"):
        base = next((p for p in parsed if p["name"] == f"baseline-{scale}"), None)
        fan = next((p for p in parsed if p["name"] == f"fanin-{scale}"), None)
        if not base or not fan:
            continue
        print(f"\n## {scale} — baseline vs fan-in (3 steps each)\n")
        print(f"Experiment 状态: baseline {'OK' if base['finished'] else 'INCOMPLETE'} ({len(base['steps'])} steps emitted); "
              f"fan-in {'OK' if fan['finished'] else 'INCOMPLETE'} ({len(fan['steps'])} steps emitted)")
        if base["errors"] or fan["errors"]:
            print(f"  baseline errors: {base['errors']}")
            print(f"  fan-in errors:   {fan['errors']}")
        print()
        print("| step | config    | step_s | gen_s | gs_max | gs_min | old_lp | upd_act | rsp_mean | rsp_clip |")
        print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for n in range(1, 4):
            for label, run in (("base", base), ("fan", fan)):
                s = next((x for x in run["steps"] if x["_step_num"] == n), None)
                if s is None:
                    print(f"|  {n} | {label}       |   -|   -|   -|   -|   -|   -|   -|   -|")
                    continue
                print(f"|  {n} | {label:<8} |{fmt_td(s.get('timing_s/step'))}|"
                      f"{fmt_td(s.get('timing_s/gen'))}|"
                      f"{fmt_td(s.get('timing_s/generation_timing/max'))}|"
                      f"{fmt_td(s.get('timing_s/generation_timing/min'))}|"
                      f"{fmt_td(s.get('timing_s/old_log_prob'))}|"
                      f"{fmt_td(s.get('timing_s/update_actor'))}|"
                      f"{s.get('response_length/mean', 0):8.0f} |"
                      f"{s.get('response_length/clip_ratio', 0)*100:6.1f}% |")
        # Aggregate + delta
        def mean(lst):
            return sum(lst) / len(lst) if lst else None
        def mean_metric(run, key):
            vals = [s[key] for s in run["steps"] if key in s]
            return mean(vals)
        def delta(b, f):
            if b is None or f is None or b == 0:
                return "  -"
            return f"{(f-b)/b*100:+.2f}%"
        bstep = mean_metric(base, "timing_s/step")
        fstep = mean_metric(fan, "timing_s/step")
        bgen = mean_metric(base, "timing_s/gen")
        fgen = mean_metric(fan, "timing_s/gen")
        bgsm = mean_metric(base, "timing_s/generation_timing/max")
        fgsm = mean_metric(fan, "timing_s/generation_timing/max")
        bolp = mean_metric(base, "timing_s/old_log_prob")
        folp = mean_metric(fan, "timing_s/old_log_prob")
        buact = mean_metric(base, "timing_s/update_actor")
        fuact = mean_metric(fan, "timing_s/update_actor")
        brsm = mean_metric(base, "response_length/mean")
        frsm = mean_metric(fan, "response_length/mean")
        print()
        print(f"**Aggregates** (3-step mean):")
        print(f"  step_time:           base {fmt_td(bstep)}s  fan {fmt_td(fstep)}s  Δ {delta(bstep, fstep)}")
        print(f"  gen_time:            base {fmt_td(bgen)}s  fan {fmt_td(fgen)}s  Δ {delta(bgen, fgen)}")
        print(f"  gen_max (worst rank):base {fmt_td(bgsm)}s  fan {fmt_td(fgsm)}s  Δ {delta(bgsm, fgsm)}")
        print(f"  old_log_prob:        base {fmt_td(bolp)}s  fan {fmt_td(folp)}s  Δ {delta(bolp, folp)}")
        print(f"  update_actor:        base {fmt_td(buact)}s  fan {fmt_td(fuact)}s  Δ {delta(buact, fuact)}")
        print(f"  response_length/mean:base {brsm or 0:8.0f}  fan {frsm or 0:8.0f}  Δ {delta(brsm, frsm)}")
        # Fan-in event summary
        print()
        print(f"**Fan-in events** (Exp 2/4 only):")
        beg_events = [e for e in fan["switch_events"] if e["kind"] == "begin"]
        if not beg_events:
            print("  (no switch events parsed)")
        else:
            # Only count per wall-clock time, dedup across 4 ranks
            seen = set()
            unique_begins = []
            for e in beg_events:
                key = (e["ts_str"], e["from"], e["to"])
                if key not in seen:
                    seen.add(key)
                    unique_begins.append(e)
            print(f"  total unique topology switches: {len(unique_begins)}")
            dp_to_tp = sum(1 for e in unique_begins if e["from"] == "dp" and e["to"] == "tp")
            tp_to_dp = sum(1 for e in unique_begins if e["from"] == "tp" and e["to"] == "dp")
            print(f"    dp→tp (fan-in fire):    {dp_to_tp}")
            print(f"    tp→dp (switch-back):    {tp_to_dp}")
            for e in unique_begins[:8]:
                print(f"    {e['ts_str']}  rank{e['rank']}  {e['from']}->{e['to']}  strategy={e['strategy']}")


def build_chrome_trace(parsed):
    """Build a merged chrome trace JSON across all 4 experiments.

    Each experiment is a "process" (pid). Each rank is a "thread" (tid 0..3).
    Each step becomes 4 phase boxes (rollout, old_log_prob, update_actor, ref).
    Topology switches become instant markers.
    Time origin: each experiment's clock resets to 0 at step 1 start.
    """
    events = []
    for exp in parsed:
        pid = exp["pid"]
        events.append({"ph": "M", "name": "process_name", "pid": pid, "tid": 0,
                       "args": {"name": exp["label"]}})
        events.append({"ph": "M", "name": "process_sort_index", "pid": pid, "tid": 0,
                       "args": {"sort_index": pid}})
        for tid in range(4):
            events.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid,
                           "args": {"name": f"rank {tid}"}})
        # Synthesize phase boxes per step. Use cumulative wall-clock from step times.
        t_cursor_us = 0
        for s in exp["steps"]:
            step = s.get("_step_num", 1)
            gs_max = s.get("timing_s/generation_timing/max", 0)
            gs_min = s.get("timing_s/generation_timing/min", 0)
            gen_total = s.get("timing_s/gen", gs_max)
            olp = s.get("timing_s/old_log_prob", 0)
            uact = s.get("timing_s/update_actor", 0)
            step_total = s.get("timing_s/step", gen_total + olp + uact)
            # For each rank, emit:
            # - rollout block: duration = gs_max for rank 3 (worst),
            #   gs_min for rank 0 (best), proportional for ranks 1/2
            # - followed by idle gap if shorter, then old_log_prob + update_actor
            worst_dur_us = int(gen_total * 1e6)
            for tid in range(4):
                # Heuristic: rank 0 fastest (gs_min), rank 3 slowest (gs_max)
                # Linear interp for ranks 1, 2
                if gs_max == gs_min:
                    rank_gen_us = worst_dur_us
                else:
                    frac = tid / 3
                    rank_gen_us = int((gs_min + (gs_max - gs_min) * frac) * 1e6)
                events.append({
                    "ph": "X", "name": f"rollout step{step}",
                    "pid": pid, "tid": tid,
                    "ts": t_cursor_us, "dur": rank_gen_us,
                    "cat": "rollout",
                    "args": {"rank_gen_s": rank_gen_us / 1e6,
                             "rsp_mean": s.get("response_length/mean", 0)},
                })
                # Idle gap for faster ranks
                if rank_gen_us < worst_dur_us:
                    events.append({
                        "ph": "X", "name": "idle (tail gap)",
                        "pid": pid, "tid": tid,
                        "ts": t_cursor_us + rank_gen_us,
                        "dur": worst_dur_us - rank_gen_us,
                        "cat": "idle",
                    })
                # old_log_prob on all ranks after worst finishes
                olp_begin = t_cursor_us + worst_dur_us
                events.append({
                    "ph": "X", "name": f"old_log_prob step{step}",
                    "pid": pid, "tid": tid,
                    "ts": olp_begin, "dur": int(olp * 1e6),
                    "cat": "old_log_prob",
                })
                uact_begin = olp_begin + int(olp * 1e6)
                events.append({
                    "ph": "X", "name": f"update_actor step{step}",
                    "pid": pid, "tid": tid,
                    "ts": uact_begin, "dur": int(uact * 1e6),
                    "cat": "update_actor",
                })
            t_cursor_us += int(step_total * 1e6)
        # Topology switch markers (for fan-in runs). We don't have exact wall-clock
        # alignment to t_cursor since scheduler timestamps are absolute; convert
        # relative to the first switch being at rollout begin of step 1.
        begin_events = [e for e in exp["switch_events"] if e["kind"] == "begin"]
        if begin_events:
            # Dedup by (ts, from, to) — all 4 ranks emit the same event
            seen = set()
            unique = []
            for e in begin_events:
                key = (e["ts_str"], e["from"], e["to"])
                if key not in seen:
                    seen.add(key)
                    unique.append(e)
            if unique:
                t0 = unique[0]["dt"]
                for e in unique:
                    dt_rel_s = (e["dt"] - t0).total_seconds()
                    events.append({
                        "ph": "i", "s": "g",
                        "name": f"{e['from']}->{e['to']}",
                        "pid": pid, "tid": 0,
                        "ts": int(dt_rel_s * 1e6),
                        "cat": "topo_switch",
                        "args": {"strategy": e.get("strategy", "?")},
                    })
    return {"traceEvents": events, "displayTimeUnit": "ms"}


def main():
    parsed = []
    for exp in EXPERIMENTS:
        p = parse_log(exp["log"])
        p.update(exp)
        parsed.append(p)
    # Print tables
    print_comparison_table(parsed)
    # Emit chrome trace
    trace = build_chrome_trace(parsed)
    out_path = Path("/tmp/sgfanin_audit_merged.json")
    out_path.write_text(json.dumps(trace))
    print(f"\nChrome trace written to {out_path} ({out_path.stat().st_size/1024:.0f} KB)")
    print("Load in https://ui.perfetto.dev to visualize.")
    # Per-experiment copies
    for exp in EXPERIMENTS:
        if exp["profiling_dir"].exists():
            exp_trace = {"traceEvents": [e for e in trace["traceEvents"]
                                         if e.get("pid") == exp["pid"] or e.get("ph") == "M"],
                         "displayTimeUnit": "ms"}
            dst = exp["profiling_dir"] / "trace.json"
            dst.write_text(json.dumps(exp_trace))


if __name__ == "__main__":
    main()
