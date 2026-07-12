#!/usr/bin/env python3
"""
Leave-one-family-out generalization experiment.

For each DGA family, remove it entirely from training and measure recall on that
family alone. This exposes how far the model drops on families it never saw.
Outputs a per-family recall bar chart with the normal test-set recall as a
reference line.

Usage: python src/leave_one_out.py --data results/processed.csv --epochs 4 \
        --benign-cap 40000 --other-cap 6000
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import encode_domains, load_processed, ALL_FAMILIES, DICT_FAMILIES
from train import train_lstm, lstm_predict_proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/processed.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--benign-cap", type=int, default=0,
                    help="benign cap per fold (0 = use all); lower it to speed up on CPU")
    ap.add_argument("--other-cap", type=int, default=0,
                    help="per-family cap for the other DGA families (0 = use all)")
    ap.add_argument("--ref-recall", type=float, default=None,
                    help="normal test-set recall reference line; read from metrics_main.json if omitted")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_processed(args.data)
    df = df.reset_index(drop=True)
    rng = np.random.RandomState(42)

    # Reference line: recall on the normal (mixed-family) test set.
    ref = args.ref_recall
    if ref is None:
        mp = os.path.join(args.outdir, "metrics_main.json")
        if os.path.exists(mp):
            ref = json.load(open(mp))["lstm"]["recall_pos"]

    benign_all = df[df.label == 0]
    results = {}
    for fam in ALL_FAMILIES:
        fam_rows = df[df.family == fam]
        if len(fam_rows) == 0:
            continue
        other_dga = df[(df.label == 1) & (df.family != fam)]
        if args.other_cap > 0:
            other_dga = (other_dga.groupby("family", group_keys=False)
                         .apply(lambda g: g.sample(min(len(g), args.other_cap),
                                                   random_state=42)))
        ben = benign_all
        if args.benign_cap > 0 and len(ben) > args.benign_cap:
            ben = ben.sample(args.benign_cap, random_state=42)

        train_rows = np.concatenate([ben.index.values, other_dga.index.values])
        rng.shuffle(train_rows)
        Xtr = encode_domains(df.loc[train_rows, "sld"].tolist())
        ytr = df.loc[train_rows, "label"].values
        Xte = encode_domains(fam_rows["sld"].tolist())

        print(f"[LOO] hold out {fam}: train={len(train_rows)} test(fam)={len(fam_rows)}")
        model, _ = train_lstm(Xtr, ytr, epochs=args.epochs, verbose=True,
                              log=lambda m: print("   " + m))
        p = lstm_predict_proba(model, Xte)
        recall = float((p >= 0.5).mean())  # all held-out samples are positive
        results[fam] = round(recall, 4)
        print(f"   -> unseen recall({fam}) = {recall:.4f}")
        # Checkpoint after every fold.
        with open(os.path.join(args.outdir, "loo_results.json"), "w", encoding="utf-8") as f:
            json.dump({"ref_test_recall": ref, "unseen_recall": results,
                       "dict_families": DICT_FAMILIES}, f, ensure_ascii=False, indent=2)

    # Bar chart
    fams = list(results.keys())
    vals = [results[f] for f in fams]
    colors = ["#d62728" if f in DICT_FAMILIES else "#1f77b4" for f in fams]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(fams, vals, color=colors)
    if ref is not None:
        plt.axhline(ref, color="green", linestyle="--",
                    label=f"normal test-set recall = {ref:.2f}")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                 ha="center", fontsize=8)
    plt.ylabel("Recall on UNSEEN family"); plt.ylim(0, 1.05)
    plt.title("Leave-one-family-out: recall on unseen families\n(red = dictionary/word-based, blue = arithmetic)")
    plt.legend(); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "loo_recall.png"), dpi=120); plt.close()

    dict_avg = np.mean([results[f] for f in DICT_FAMILIES if f in results])
    rand_avg = np.mean([results[f] for f in ALL_FAMILIES
                        if f in results and f not in DICT_FAMILIES])
    summary = {"ref_test_recall": ref, "unseen_recall": results,
               "dict_family_avg_recall": round(float(dict_avg), 4),
               "random_family_avg_recall": round(float(rand_avg), 4)}
    with open(os.path.join(args.outdir, "loo_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n=== LOO summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
