#!/usr/bin/env bash
# Run training + evaluation, then leave-one-family-out. Checkpoint after each stage.
set -e
cd "$(dirname "$0")/.."
PY=./venv/bin/python
PROG=.progress/build.jsonl

echo '{"step":"train-start"}' >> $PROG
$PY src/train.py --data results/processed.csv --epochs 8 2>&1 | tee results/log_train.txt
echo '{"step":"train-done","next":"loo"}' >> $PROG

echo '{"step":"loo-start"}' >> $PROG
$PY src/leave_one_out.py --data results/processed.csv --epochs 4 \
    --benign-cap 40000 --other-cap 5000 2>&1 | tee results/log_loo.txt
echo '{"step":"loo-done","next":"llm+report"}' >> $PROG
echo "ALL_EXPERIMENTS_DONE"
