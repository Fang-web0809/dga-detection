#!/usr/bin/env python3
"""
llm_baseline.py — 步驟 6 的 LLM few-shot 對照組。

用 Claude Code 的無頭模式(`claude -p`)判斷網域是否為 DGA 生成。
走使用者的訂閱登入(與互動模式同一組驗證),**不需要 API key、不需要 Ollama**。

前提:
    - VM 上已安裝並登入 Claude Code(互動式 `claude` 能正常回應即可)。
    - `claude` 指令在 PATH 中。

用法範例:
    python src/llm_baseline.py \
        --input results/llm_sample.csv \
        --outdir results \
        --model haiku \
        --batch-size 50 \
        --latency-n 30

輸入 CSV 需含欄位:
    - 網域欄(預設 `sld`,與 LSTM 看到的表示一致;可用 --domain-col 改)
    - `label`  (1=DGA, 0=benign)
    - `family` (家族名;benign 樣本可填 "benign")

輸出:
    - <outdir>/llm_predictions.csv   逐筆 domain / true_label / family / pred / raw
    - <outdir>/llm_metrics.json      accuracy / precision / recall / f1、每家族 recall、
                                     總成本、平均延遲、使用模型等

設計重點:
    - 準確率用「一次批次」跑(把一批網域一起丟給模型),快又省額度。
    - 延遲另在小樣本上「逐筆」量(每筆一次 claude -p),因為要 per-domain 數字。
      注意:逐筆呼叫含 CLI 啟動開銷,量到的延遲偏高,報告請標註為近似值。
    - 不使用 --bare(bare 會跳過 OAuth 而強制要 API key)。改在中性工作目錄執行,
      避免自動載入專案 CLAUDE.md 進 context。
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

# few-shot 範例(與交接文件的 prompt 一致)
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
    """呼叫 claude -p,回傳 (result_text, cost_usd, usage_dict)。失敗則丟例外。"""
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--max-turns", "1"]
    if model:
        cmd += ["--model", model]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=workdir
        )
    except FileNotFoundError:
        sys.exit("找不到 `claude` 指令。請先在此 VM 安裝並登入 Claude Code。")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"claude -p 逾時({timeout}s)")
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p 失敗 (code {proc.returncode}): {proc.stderr[:500]}")
    try:
        obj = json.loads(proc.stdout)
    except json.JSONDecodeError:
        # 萬一不是 JSON,就把原始輸出當結果、成本記 None
        return proc.stdout.strip(), None, {}
    result = obj.get("result", "")
    cost = obj.get("total_cost_usd")
    usage = obj.get("usage", {}) or {}
    return result, cost, usage


def parse_verdicts(text, n):
    """
    從模型回覆解析 n 筆判定。預期每行格式 `<編號>. DGA` 或 `<編號>. BENIGN`。
    回傳長度 n 的 list,元素為 1(DGA)/0(BENIGN)/None(無法解析)。
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
    """一次判斷一批網域,回傳與 domains 等長的預測 list(1/0/None)。"""
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
    ap.add_argument("--input", required=True, help="抽樣後的測試 CSV")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--domain-col", default="sld",
                    help="要送給 LLM 的網域欄(預設 sld,與 LSTM 一致)")
    ap.add_argument("--model", default=None,
                    help="Claude 模型別名或名稱(如 haiku);省略則用 Claude Code 預設")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--latency-n", type=int, default=30,
                    help="逐筆量延遲的樣本數(0 表示不量延遲)")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    col = args.domain_col if args.domain_col in df.columns else "domain"
    if col not in df.columns:
        sys.exit(f"輸入 CSV 找不到網域欄('{args.domain_col}' 或 'domain')")
    for req in ("label", "family"):
        if req not in df.columns:
            sys.exit(f"輸入 CSV 缺少必要欄位:{req}")

    domains = df[col].astype(str).tolist()
    y_true = df["label"].astype(int).tolist()
    families = df["family"].astype(str).tolist()
    print(f"讀入 {len(domains)} 筆,網域欄='{col}',模型={args.model or 'Claude Code 預設'}")

    # 中性工作目錄:避免 claude -p 自動載入本專案 CLAUDE.md 進 context
    workdir = tempfile.mkdtemp(prefix="llm_baseline_")

    # ---- 批次分類(準確率)----
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
        print(f"  批次 {b+1}/{n_batches} 完成({done}/{len(domains)})")

    # 無法解析的預設當 BENIGN(0),並記錄數量
    unresolved = sum(1 for x in preds if x is None)
    preds = [0 if x is None else x for x in preds]

    # ---- 逐筆量延遲(小樣本)----
    latencies = []
    if args.latency_n > 0:
        sub = domains[:args.latency_n]
        for d in sub:
            t0 = time.perf_counter()
            try:
                classify_batch([d], args.model, workdir)
            except Exception as e:
                print(f"  延遲量測略過一筆:{e}")
                continue
            latencies.append(time.perf_counter() - t0)
    avg_latency = sum(latencies) / len(latencies) if latencies else None

    # ---- 指標 ----
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    # 每家族 recall(只看 DGA 樣本)
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
        "latency_note": "逐筆 claude -p 含 CLI 啟動開銷,為近似上界",
    }

    # ---- 輸出 ----
    import os
    os.makedirs(args.outdir, exist_ok=True)
    pred_df = df.copy()
    pred_df["pred"] = preds
    pred_df.to_csv(os.path.join(args.outdir, "llm_predictions.csv"), index=False)
    with open(os.path.join(args.outdir, "llm_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== LLM 對照組結果 ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if unresolved:
        print(f"\n注意:{unresolved} 筆模型回覆無法解析,已當 BENIGN 計。"
              f"若比例偏高,考慮縮小 batch-size 或加強 prompt 格式約束。")


if __name__ == "__main__":
    main()
