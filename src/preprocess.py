#!/usr/bin/env python3
"""
Build the labeled dataset from raw DGA family lists and Tranco.

Reads the 11 DGA family files and the Tranco top-1M list, merges them, keeps the
family label through sampling, reduces each domain to its SLD, drops short SLDs,
deduplicates before splitting (to avoid leakage), and writes a stratified
train/test split to results/processed.csv (columns: sld, label, family, split).

Usage:
    python src/preprocess.py --dga-dir data/dga --tranco data/tranco.csv \
        --out results/processed.csv --benign-n 300000 --family-cap 10000
"""
import argparse
import glob
import os
import sys

import pandas as pd
import tldextract
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from common import ALL_FAMILIES


def get_extractor():
    """Prefer the online/cached suffix list; fall back to the bundled snapshot if offline."""
    try:
        ext = tldextract.TLDExtract()
        ext("example.com")  # force the list to load
        return ext
    except Exception:
        print("  [warn] could not fetch the online suffix list; using the offline snapshot")
        return tldextract.TLDExtract(suffix_list_urls=())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dga-dir", default="data/dga")
    ap.add_argument("--tranco", default="data/tranco.csv")
    ap.add_argument("--out", default="results/processed.csv")
    ap.add_argument("--benign-n", type=int, default=300000)
    ap.add_argument("--family-cap", type=int, default=10000)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Malicious samples: one file per family, family = filename without extension.
    dga_rows = []
    for path in sorted(glob.glob(os.path.join(args.dga_dir, "*.txt"))):
        fam = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                d = line.strip()
                if d:
                    dga_rows.append((d, 1, fam))
    dga = pd.DataFrame(dga_rows, columns=["domain", "label", "family"])
    print(f"loaded {len(dga)} DGA domains; families: {sorted(dga.family.unique())}")

    # pandas >=2.x drops the group key here; select columns explicitly (and
    # group_keys=False) so `family` survives the per-group sample.
    dga_s = (dga.groupby("family", group_keys=False)[["domain", "label", "family"]]
                .apply(lambda g: g.sample(min(len(g), args.family_cap),
                                          random_state=args.seed)))
    print(f"sampled to {len(dga_s)} DGA domains (cap {args.family_cap}/family)")

    # Benign samples: Tranco (rank, domain), no header row.
    tr = pd.read_csv(args.tranco, names=["rank", "domain"])
    tr = tr.sample(min(len(tr), args.benign_n), random_state=args.seed)
    benign = pd.DataFrame({"domain": tr["domain"].astype(str).values,
                           "label": 0, "family": "benign"})
    print(f"sampled {len(benign)} benign domains")

    df = pd.concat([dga_s[["domain", "label", "family"]], benign],
                   ignore_index=True)

    # Reduce every domain to its SLD; DGA randomness lives there, and keeping the
    # TLD would let the model cheat on the generalization test.
    ext = get_extractor()
    df["sld"] = df["domain"].map(lambda d: ext(str(d)).domain.lower())

    before = len(df)
    df = df[df["sld"].str.len() >= args.min_len].copy()
    print(f"dropped SLDs shorter than {args.min_len}: {before} -> {len(df)}")

    # Guard against families whose randomness hides in the subdomain: their SLD
    # collapses to a near-constant value and they should be excluded (e.g. symmi).
    for fam in ALL_FAMILIES:
        sub = df[df.family == fam]
        if len(sub) and sub["sld"].nunique() <= max(2, len(sub) * 0.01):
            print(f"  [warn] family {fam} has near-constant SLDs "
                  f"(nunique={sub['sld'].nunique()}); randomness likely in the subdomain")

    # Deduplicate after merging (then split) so the same SLD can't leak across sets.
    df = df.drop_duplicates(subset="sld").reset_index(drop=True)
    print(f"after dedup: {len(df)} (benign {(df.label==0).sum()} / dga {(df.label==1).sum()})")

    assert df.loc[df.label == 1, "family"].notna().all(), "family label lost during sampling"

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=args.seed)
    df.loc[train_df.index, "split"] = "train"
    df.loc[test_df.index, "split"] = "test"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df[["sld", "label", "family", "split"]].to_csv(args.out, index=False)
    print(f"wrote {args.out}  train={len(train_df)} test={len(test_df)}")

    fam_counts = df[df.label == 1].groupby("family").size().to_dict()
    print("per-family counts (after dedup):", fam_counts)


if __name__ == "__main__":
    main()
