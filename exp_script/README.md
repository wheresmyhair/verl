# exp_script

Paper-ready experiment scripts. Flat layout — four scripts at the root,
one for each arm of two 2-way comparisons.

## Layout

```
exp_script/
├── common.sh             shared env (model, batch, output paths)
├── idea1_fanin.sh        Idea 1 treatment: rlpipe SGLang fork + fan-in
├── idea1_baseline.sh     Idea 1 control:   vanilla SGLang DP=4
├── idea2_fused.sh        Idea 2 treatment: torch_pp + fused_forward=True
├── idea2_baseline.sh     Idea 2 control:   torch_pp + fused_forward=False
└── functionalities/      staged / smoke / one-off scripts (not for paper)
```

Training side is identical across all four scripts (torch_pp PP=4,
`actor.fused_forward=True` for idea1 pair, fused on/off for idea2 pair).

## Running

```bash
bash exp_script/idea1_baseline.sh
SEED=7 TOTAL_STEPS=1 bash exp_script/idea1_fanin.sh   # smoke run
```

Every script:
- sources `common.sh` for shared defaults (override via env vars),
- writes traces and logs under `$PROFILING_ROOT/$EXP_NAME/seed_$SEED/`,
- uses `trainer.project_name=rlpipe`, `experiment_name=${EXP_NAME}_seed${SEED}`.

## Paper comparisons

| Comparison         | Treatment script      | Baseline script         | Delta isolates |
|--------------------|-----------------------|-------------------------|----------------|
| Idea 1 (rollout)   | `idea1_fanin.sh`      | `idea1_baseline.sh`     | DP→TP fan-in   |
| Idea 2 (training)  | `idea2_fused.sh`      | `idea2_baseline.sh`     | fused forward  |

Baseline = vanilla verl run (no fork, no fused). Each pair differs in
exactly one knob; the other (rollout vs. training) is held fixed.

## Acceptance

A script is runnable when it completes `TOTAL_STEPS=5` without OOM at
`BATCH=16`, `N_SAMPLES=16`, `MAX_RESP=16384` on 4×A100-80G.

## Prerequisites

- `idea1_fanin.sh` requires the rlpipe SGLang fork
  (`/home/user/rlpipe/sglang-fork`, branch `rlpipe/dynamic-tp`) installed.
- The other three scripts run on stock SGLang.
