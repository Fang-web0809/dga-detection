#!/usr/bin/env python3
"""
LLM few-shot baseline (step 6).

Classifies domains as DGA or benign via Claude Code's headless mode (`claude -p`),
using the user's subscription login (same auth as interactive mode) -- no API key
and no Ollama required.

Requirements:
    - Claude Code installed and logged in (interactive `claude` responds).
    - `claude` on PATH.

Example:
    python src/llm_baseline.py \
        --input results/llm_sample.csv \
        --outdir results \
        --model haiku \
        --batch-size 50 \
        --latency-n 30

Input CSV columns:
    - a domain column (default `sld`, matching what the LSTM sees; override with --domain-col)
    - `label`  (1 = DGA, 0 = benign)
    - `family` (family name; benign rows can use "benign")

Outputs:
    - <outdir>/llm_predictions.csv   per-row domain / true_label / family / pred / raw
    - <outdir>/llm_metrics.json      accuracy / precision / recall / f1, per-family recall,
                                     total cost, average latency, model used, etc.

Notes:
    - Accuracy is measured in batches (many domains per call) to save time and quota.
    - Latency is measured separately, one domain per call, since we need a per-domain
      number. Single calls include CLI start-up overhead, so treat it as an upper bound.
    - --bare is avoided: it skips OAuth and forces an API key. Runs from a neutral
      working directory so the project CLAUDE.md is not auto-loaded into context.
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# Few-shot prompt (kept in Chinese: this is the exact prompt used to produce the
# reported results; changing it would change model behavior).
SYSTEM_PROMPT = (
    "你是資安分析師,負責判斷網域是否為 DGA(域名生成演算法)產生。"
    "DGA 網域通常缺乏可讀性、字元組合異常。嚴格依指示格式作答,不要多餘說明。"
)
FEWSHOT = (
    "判斷規則參考範例:\n"
    "正常:google、wikipedia、shopee、momoshop\n"
    "DGA:kq3vz8xw1、xjkw92mfp、qzvbnw31k\n"
)


def run_claude(prompt, model=None, timeout=120, workdir=None):
    """Call `claude -p` and return (result_text, cost_usd, usage_dict). Raises on failure."""
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--max-turns", "1"]
    if model:
        cmd += ["--model", model]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=workdir
        )
    except FileNotFoundError:
        sys.exit("`claude` not found. Install and log in to Claude Code on this machine first.")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"claude -p timed out ({timeout}s)")
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p failed (code {proc.returncode}): {proc.stderr[:500]}")
    try:
        obj = json.loads(proc.stdout)
    except json.JSONDecodeError:
        # Not JSON: return the raw text and no cost.
        return proc.stdout.strip(), None, {}
    result = obj.get("result", "")
    cost = obj.get("total_cost_usd")
    usage = obj.get("usage", {}) or {}
    return result, cost, usage


def parse_verdicts(text, n):
    """Parse n verdicts from the reply, expecting `<index>. DGA` / `<index>. BENIGN` per line.

    Returns a list of length n with 1 (DGA), 0 (BENIGN) or None (unparseable).
    """
    verdicts = [None] * n
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)[\.\):\s]+.*?\b(DGA|BENIGN)\b", line, re.IGNORECASE)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if 0 <= idx < n:
            verdicts[idx] = 1 if m.group(2).upper() == "DGA" else 0
    return verdicts


def classify_batch(domains, model, workdir):
    """Classify one batch of domains; returns a prediction list aligned with `domains` (1/0/None)."""
    numbered = "\n".join(f"{i+1}. {d}" for i, d in enumerate(domains))
    prompt = (
        f"{SYSTEM_PROMPT}\n\n{FEWSHOT}\n"
        f"以下每行一個網域。請逐一判定是否為 DGA 產生,"
        f"回覆格式嚴格為每行 `<編號>. DGA` 或 `<編號>. BENIGN`,共 {len(domains)} 行,"
        f"不要輸出其他文字:\n{numbered}"
    )
    result, cost, usage = run_claude(prompt, model=model, workdir=workdir)
    preds = parse_verdicts(result, len(domains))
    return preds, cost, usage, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="sampled test CSV")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--domain-col", default="sld",
                    help="domain column to send to the LLM (default sld, matching the LSTM)")
    ap.add_argument("--model", default=None,
                    help="Claude model alias/name (e.g. haiku); omit to use the Claude Code default")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--latency-n", type=int, default=30,
                    help="number of single-call latency samples (0 = skip latency)")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    col = args.domain_col if args.domain_col in df.columns else "domain"
    if col not in df.columns:
        sys.exit(f"domain column not found in input CSV ('{args.domain_col}' or 'domain')")
    for req in ("label", "family"):
        if req not in df.columns:
            sys.exit(f"input CSV missing required column: {req}")

    domains = df[col].astype(str).tolist()
    y_true = df["label"].astype(int).tolist()
    families = df["family"].astype(str).tolist()
    print(f"loaded {len(domains)} rows, domain col='{col}', model={args.model or 'Claude Code default'}")

    # Neutral working directory so claude -p does not auto-load this project's CLAUDE.md.
    workdir = tempfile.mkdtemp(prefix="llm_baseline_")

    # --- batch classification (accuracy) ---
    preds, total_cost, total_in, total_out = [], 0.0, 0, 0
    n_batches = (len(domains) + args.batch_size - 1) // args.batch_size
    for b in range(n_batches):
        chunk = domains[b * args.batch_size:(b + 1) * args.batch_size]
        p, cost, usage, _ = classify_batch(chunk, args.model, workdir)
        preds.extend(p)
        if cost:
            total_cost += cost
        total_in += usage.get("input_tokens", 0) or 0
        total_out += usage.get("output_tokens", 0) or 0
        done = min((b + 1) * args.batch_size, len(domains))
        print(f"  batch {b+1}/{n_batches} done ({done}/{len(domains)})")

    # Unparseable verdicts default to BENIGN (0); record how many.
    unresolved = sum(1 for x in preds if x is None)
    preds = [0 if x is None else x for x in preds]

    # --- per-domain latency on a small sample ---
    latencies = []
    if args.latency_n > 0:
        sub = domains[:args.latency_n]
        for d in sub:
            t0 = time.perf_counter()
            try:
                classify_batch([d], args.model, workdir)
            except Exception as e:
                print(f"  skipped a latency sample: {e}")
                continue
            latencies.append(time.perf_counter() - t0)
    avg_latency = sum(latencies) / len(latencies) if latencies else None

    # --- metrics ---
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    # Per-family recall (over DGA samples only).
    fam_hit, fam_tot = defaultdict(int), defaultdict(int)
    for yt, yp, fam in zip(y_true, preds, families):
        if yt == 1:
            fam_tot[fam] += 1
            fam_hit[fam] += int(yp == 1)
    per_family_recall = {f: fam_hit[f] / fam_tot[f] for f in fam_tot}

    metrics = {
        "model": args.model or "claude-code-default",
        "n_samples": len(domains),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "unresolved_count": unresolved,
        "per_family_recall": {k: round(v, 4) for k, v in sorted(per_family_recall.items())},
        "total_cost_usd": round(total_cost, 6) if total_cost else None,
        "cost_per_1000_usd": round(total_cost / len(domains) * 1000, 4) if total_cost else None,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "avg_latency_s_per_domain": round(avg_latency, 3) if avg_latency else None,
        "latency_note": "single-call claude -p includes CLI start-up overhead; treat as an upper bound",
    }

    # --- outputs ---
    import os
    os.makedirs(args.outdir, exist_ok=True)
    pred_df = df.copy()
    pred_df["pred"] = preds
    pred_df.to_csv(os.path.join(args.outdir, "llm_predictions.csv"), index=False)
    with open(os.path.join(args.outdir, "llm_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== LLM baseline ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if unresolved:
        print(f"\nnote: {unresolved} replies were unparseable and counted as BENIGN. "
              f"If this fraction is high, reduce batch-size or tighten the prompt format.")


if __name__ == "__main__":
    main()
