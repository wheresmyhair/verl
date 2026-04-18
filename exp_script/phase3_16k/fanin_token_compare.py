#!/usr/bin/env python3
"""Compare baseline vs fan-in per-request token dumps.

Usage:
    python fanin_token_compare.py <baseline_dir> <fanin_dir>

Loads baseline.json and fanin.json, compares per-request:
  - output_ids: the IDs fan-in merges (partial + new)
  - logprob_ids: the IDs _post_process_outputs actually uses
  - partial_ids: fan-in's pre-abort tokens (for stragglers)
"""
import json
import sys
from pathlib import Path


def load(p):
    return json.load(open(p))


def first_diverge(a, b):
    """Return index of first mismatch, or min(len(a),len(b)) if prefix matches."""
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b))


def main():
    base_dir, fanin_dir = Path(sys.argv[1]), Path(sys.argv[2])
    # Baseline may be split per-rank (stock DP: baseline_rank0.json etc)
    # or single file (dynamic-tp: baseline.json). Load all and merge.
    base_records = []
    for p in sorted(base_dir.glob("baseline*.json")):
        base_records.extend(load(p))
    base = {r["request_index"]: r for r in base_records}
    fanin = {r["request_index"]: r for r in load(fanin_dir / "fanin.json")}

    print(f"{'idx':>3} {'mode':<18} {'base_oid':>8} {'fan_oid':>8} {'base_lp':>8} {'fan_lp':>8} "
          f"{'partial':>7} {'oid_div':>7} {'lp_div':>6} {'partial_match':>13}")
    print("-" * 110)

    for i in sorted(set(base.keys()) | set(fanin.keys())):
        b = base.get(i)
        f = fanin.get(i)
        if not b or not f:
            print(f"{i:>3} MISSING in {'baseline' if not b else 'fanin'}")
            continue

        b_oid = b["output_ids"]
        f_oid = f["output_ids"]
        b_lp = b["logprob_ids"]
        f_lp = f["logprob_ids"]

        oid_div = first_diverge(b_oid, f_oid)
        lp_div = first_diverge(b_lp, f_lp)

        partial_len = f.get("partial_len", "-")
        partial_match = "-"
        if f["mode"] == "fanin_straggler":
            partial = f.get("partial_ids", [])
            # Does partial match the baseline output prefix?
            pm = first_diverge(partial, b_oid)
            partial_match = f"{'YES' if pm == len(partial) else 'NO@'+str(pm)}"

        print(f"{i:>3} {f['mode']:<18} {len(b_oid):>8} {len(f_oid):>8} {len(b_lp):>8} {len(f_lp):>8} "
              f"{str(partial_len):>7} {oid_div:>7} {lp_div:>6} {str(partial_match):>13}")

    # Summary
    print()
    stragglers = [f for f in fanin.values() if f["mode"] == "fanin_straggler"]
    completed = [f for f in fanin.values() if f["mode"] == "fanin_completed"]
    print(f"Completed in DP (no abort): {len(completed)}")
    print(f"Stragglers (aborted + reprefilled): {len(stragglers)}")

    if stragglers:
        # Check hypothesis 3: logprob_ids vs output_ids mismatch in fan-in
        lp_oid_mismatch = 0
        for s in stragglers:
            if s["logprob_ids"] != s["output_ids"][:len(s["logprob_ids"])]:
                lp_oid_mismatch += 1
        print(f"\nHypothesis 3 (logprob_ids ≠ output_ids prefix in fan-in stragglers): "
              f"{lp_oid_mismatch}/{len(stragglers)} mismatched")

        # Check if logprob_ids length = stage3_new_len (= only post-reprefill)
        lp_eq_stage3 = sum(1 for s in stragglers if len(s["logprob_ids"]) == s.get("stage3_new_len", -1))
        print(f"logprob_ids length == stage3_new_len: {lp_eq_stage3}/{len(stragglers)}")
        print("  (If all match → logprob_ids has ONLY post-reprefill tokens, missing partial)")

        # Check if partial matches baseline prefix
        partial_ok = sum(1 for s in stragglers
                        if s["request_index"] in base and
                        first_diverge(s.get("partial_ids",[]), base[s["request_index"]]["output_ids"])
                           == len(s.get("partial_ids",[])))
        print(f"\nPartial IDs match baseline prefix: {partial_ok}/{len(stragglers)}")
        print("  (If all match → DP decode was numerically identical before abort)")

    # Decode a few tokens around divergence for one straggler
    if stragglers:
        s = stragglers[0]
        b = base[s["request_index"]]
        p = s.get("partial_ids", [])
        print(f"\n--- Example straggler: request {s['request_index']} ---")
        print(f"  baseline output_ids length: {len(b['output_ids'])}")
        print(f"  fanin output_ids length:    {len(s['output_ids'])}")
        print(f"  fanin logprob_ids length:   {len(s['logprob_ids'])}")
        print(f"  partial_ids length:         {len(p)}")
        print(f"  stage3_new_ids length:      {s.get('stage3_new_len', '?')}")
        div = first_diverge(b["output_ids"], s["output_ids"])
        print(f"  output_ids first diverge:   position {div}")
        if div < len(b["output_ids"]) and div < len(s["output_ids"]):
            print(f"    baseline[{div}] = {b['output_ids'][div]}")
            print(f"    fanin[{div}]    = {s['output_ids'][div]}")


if __name__ == "__main__":
    main()
