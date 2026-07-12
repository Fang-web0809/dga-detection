#!/usr/bin/env bash
# 步驟 3-5 編排:主訓練+評估 → leave-one-family-out。每階段完成寫進度檔。
set -e
cd "$(dirname "$0")/.."
PY=./venv/bin/python
PROG=.progress/build.jsonl

echo '{"ts":"run","step":"3-4-start","done":"開始主訓練+評估"}' >> $PROG
$PY src/train.py --data results/processed.csv --epochs 8 2>&1 | tee results/log_train.txt
echo '{"ts":"run","step":"3-4-done","done":"主訓練+評估完成(metrics_main.json 等)","next":"LOO"}' >> $PROG

echo '{"ts":"run","step":"5-start","done":"開始 leave-one-family-out"}' >> $PROG
$PY src/leave_one_out.py --data results/processed.csv --epochs 4 \
    --benign-cap 40000 --other-cap 5000 2>&1 | tee results/log_loo.txt
echo '{"ts":"run","step":"5-done","done":"LOO 完成(loo_results.json, loo_recall.png)","next":"LLM+report"}' >> $PROG
echo "ALL_EXPERIMENTS_DONE"
