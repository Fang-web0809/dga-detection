#!/usr/bin/env python3
"""Sample the test set for the LLM baseline: n_per_family DGA per family plus a batch of benign."""
import argparse, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import load_processed, ALL_FAMILIES

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="results/processed.csv")
ap.add_argument("--out", default="results/llm_sample.csv")
ap.add_argument("--per-family", type=int, default=15)
ap.add_argument("--benign", type=int, default=35)
args = ap.parse_args()

df = load_processed(args.data)
te = df[df.split == "test"]
parts = []
for fam in ALL_FAMILIES:
    sub = te[(te.label == 1) & (te.family == fam)]
    if len(sub):
        parts.append(sub.sample(min(len(sub), args.per_family), random_state=42))
ben = te[te.label == 0].sample(min((te.label == 0).sum(), args.benign), random_state=42)
parts.append(ben)
out = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42)  # shuffle
out[["sld", "label", "family"]].to_csv(args.out, index=False)
print(f"wrote {args.out}  {len(out)} rows (DGA {int((out.label==1).sum())} / benign {int((out.label==0).sum())})")
