#!/bin/bash
# Run all Phase 2 (7168) experiments sequentially
set -x
cd /home/user/rlpipe/verl

for s in exp_script/phase2_7168/{1,2,3,4,5,6,7}_*.sh; do
    echo "============================================"
    echo "Starting: $s"
    echo "Time: $(date)"
    echo "============================================"
    bash "$s"
    echo "Finished: $s at $(date)"
    echo ""
done

echo "All Phase 2 experiments completed at $(date)"
