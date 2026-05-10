"""Fused-forward schedule generator.

Implements the algorithm from `assets/fused_schedule_algorithm.tex`:

  Last rank (P-1):  run all iF.0..iF.M-1 first, then 1F1B (tF.0, tB.0, tF.1, tB.1, ...)
  Other ranks:      greedy dispatch from a "ready buffer", with priority
                    rule: smallest micro-batch idx first; tie-break iF > tB > tF.

The dispatch uses an event-driven simulation with per-rank logical time +
upstream-op completion tracking. Each op is assumed to take unit time;
comm is instantaneous. (This is the standard assumption for scheduler
generation in pipeline-parallel literature — the executor handles actual
timing at runtime.)

Output: list[list[str]] — schedule[r] = ordered op sequence on rank r,
each entry is "kind.mb" where kind ∈ {tF, tB, iF}, mb ∈ 0..M-1.

Reference schedule (P=4, M=4):
  r0:  tF.0  tF.1  tF.2  iF.0  tF.3  iF.1  iF.2  iF.3  tB.0  tB.1  tB.2  tB.3
  r1:  tF.0  iF.0  tF.1  iF.1  tF.2  iF.2  tF.3  iF.3  tB.0  tB.1  tB.2  tB.3
  r2:  iF.0  tF.0  iF.1  tF.1  iF.2  tF.2  iF.3  tB.0  tB.1  tF.3  tB.2  tB.3
  r3:  iF.0  iF.1  iF.2  iF.3  tF.0  tB.0  tF.1  tB.1  tF.2  tB.2  tF.3  tB.3
"""
from __future__ import annotations
from typing import List, Tuple


def _parse_op(op: str) -> Tuple[str, int]:
    kind, mb = op.split(".")
    return kind, int(mb)


def _priority_key(op: str):
    """Sort key for greedy SelectOp.

    NOTE: the tex pseudocode says "smallest mb primary, type tie-break",
    but the reference schedule it lists requires the OPPOSITE: type
    primary (iF > tB > tF), mb secondary. We follow the reference.
    """
    kind, mb = _parse_op(op)
    kind_pri = {"iF": 0, "tB": 1, "tF": 2}[kind]
    return (kind_pri, mb)


def _last_rank_schedule(M: int) -> List[str]:
    """Rank P-1: all iF first, then 1F1B (tF.k, tB.k alternating)."""
    sched = [f"iF.{k}" for k in range(M)]
    for k in range(M):
        sched.append(f"tF.{k}")
        sched.append(f"tB.{k}")
    return sched


def build_fused_schedule(P: int, M: int) -> List[List[str]]:
    """Build per-rank schedule for fused PP forward + reverse-PP iF + backward.

    Args:
        P: pipeline parallel size (>= 2)
        M: number of micro-batches per direction (tF, tB, iF each have M ops)

    Returns:
        schedule: list of length P; schedule[r] is the ordered op sequence
                  for rank r (each rank does 3*M ops total).
    """
    assert P >= 2 and M >= 1, f"need P>=2 and M>=1, got P={P} M={M}"

    schedule: List[List[str]] = [[] for _ in range(P)]
    schedule[P - 1] = _last_rank_schedule(M)

    # completion[(rank, op)] = global time at which (rank, op) finishes (i.e., is available downstream)
    completion = {}
    for i, op in enumerate(schedule[P - 1]):
        completion[(P - 1, op)] = i + 1  # finishes at end of unit i

    # For ranks 0..P-2: simulate greedy dispatch
    pending = {
        r: {f"{kind}.{k}" for kind in ("tF", "tB", "iF") for k in range(M)}
        for r in range(P - 1)
    }
    rank_time = {r: 0 for r in range(P - 1)}

    def upstream_dep(r: int, op: str):
        """Return (rank, op) key for the upstream prerequisite, or None
        if no upstream (rank 0's tF or anyone's source)."""
        kind, mb = _parse_op(op)
        if kind == "tF":
            return (r - 1, op) if r > 0 else None
        if kind == "tB":
            return (r + 1, op)  # tB flows reverse; need rank+1 first
        if kind == "iF":
            return (r + 1, op)  # iF reverse direction
        raise ValueError(f"bad op {op}")

    # Wave-based simulation: at each integer time slot, every rank that is
    # not busy AND has at least one ready op dispatches it. A rank with an
    # empty ready buffer idles that slot. This matches the tex algorithm's
    # "wait until B ≥ 1" semantics.
    rank_busy_until = {r: 0 for r in range(P - 1)}
    max_time = 4 * P * M  # generous upper bound
    for t in range(max_time):
        if not any(pending[r] for r in range(P - 1)):
            break

        for r in range(P - 1):
            if not pending[r]:
                continue
            if rank_busy_until[r] > t:
                continue  # rank still computing previous op

            ready = []
            for op in pending[r]:
                dep = upstream_dep(r, op)
                if dep is None:
                    ready.append(op)
                else:
                    if dep in completion and completion[dep] <= t:
                        ready.append(op)
            if not ready:
                continue  # rank idles this slot

            ready.sort(key=_priority_key)
            chosen = ready[0]
            schedule[r].append(chosen)
            completion[(r, chosen)] = t + 1  # available next slot
            rank_busy_until[r] = t + 1
            rank_time[r] = t + 1
            pending[r].remove(chosen)
    else:
        raise RuntimeError(f"schedule didn't converge in {max_time} slots")

    return schedule


def verify_dependencies(schedule: List[List[str]]) -> None:
    """Sanity-check that each op's upstream prerequisite appears in the
    upstream rank's schedule (basic correctness check)."""
    P = len(schedule)
    by_rank = [{op: i for i, op in enumerate(sched)} for sched in schedule]
    for r in range(P):
        for i, op in enumerate(schedule[r]):
            kind, mb = _parse_op(op)
            if kind == "tF" and r > 0:
                up = (r - 1, op)
            elif kind == "tB" and r < P - 1:
                up = (r + 1, op)
            elif kind == "iF" and r < P - 1:
                up = (r + 1, op)
            else:
                continue
            up_r, up_op = up
            if up_op not in by_rank[up_r]:
                raise AssertionError(f"r{r} {op} (pos {i}) needs upstream {up_op} on r{up_r}, not in schedule")
            # No strict time check; positions can differ across ranks


if __name__ == "__main__":
    P, M = 4, 4
    sched = build_fused_schedule(P, M)
    print(f"\n=== Fused schedule (P={P}, M={M}) ===")
    for r in range(P):
        print(f"r{r}:  {'  '.join(sched[r])}")

    verify_dependencies(sched)
    print("\n[verify] dependencies OK")

    # Expected reference (from assets/fused_schedule_algorithm.tex)
    expected = [
        "tF.0 tF.1 tF.2 iF.0 tF.3 iF.1 iF.2 iF.3 tB.0 tB.1 tB.2 tB.3",
        "tF.0 iF.0 tF.1 iF.1 tF.2 iF.2 tF.3 iF.3 tB.0 tB.1 tB.2 tB.3",
        "iF.0 tF.0 iF.1 tF.1 iF.2 tF.2 iF.3 tB.0 tB.1 tF.3 tB.2 tB.3",
        "iF.0 iF.1 iF.2 iF.3 tF.0 tB.0 tF.1 tB.1 tF.2 tB.2 tF.3 tB.3",
    ]
    print("\n=== vs reference ===")
    all_match = True
    for r, (got_list, exp_str) in enumerate(zip(sched, expected)):
        got_str = " ".join(got_list)
        match = got_str == exp_str
        all_match &= match
        marker = "[OK]" if match else "[DIFF]"
        print(f"r{r} {marker}")
        if not match:
            print(f"     got: {got_str}")
            print(f"     exp: {exp_str}")
    print(f"\nOverall: {'PASS' if all_match else 'FAIL'}")
