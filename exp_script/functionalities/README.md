# functionalities

Staged scripts that are NOT part of the paper comparisons. Put here:

- smoke tests for individual components (rollout-only, fused-only, etc.),
- ablations and probes (e.g. fork-without-trigger, INITIAL=tp variants),
- one-off debugging / profiling runs.

The four top-level scripts in `exp_script/` (idea1/idea2 × fanin/fused vs
baseline) stay minimal for reviewer clarity. Everything else lives here.

Naming convention: `<topic>_<variant>.sh`, e.g. `fanin_no_trigger.sh`,
`rollout_dp_only.sh`. Each script should still source `../common.sh`.
