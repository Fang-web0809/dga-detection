#!/usr/bin/env python3
"""
preprocess.py — 步驟 1c + 2。
讀 11 個 DGA 家族檔 + Tranco → 合併 → (坑一)正確保留 family → 取 SLD → 濾長度<4
→ 合併後去重(防洩漏) → stratified split。輸出 results/processed.csv(sld,label,family,split)。

用法:
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
    """優先用線上/快取的 suffix list;若不能連外則退回內建快照(交接文件第 9 節)。"""
    try:
        ext = tldextract.TLDExtract()
        ext("example.com")  # 觸發載入
        return ext
    except Exception:
        print("  [warn] 無法取得線上 suffix list,改用離線快照")
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

    # ---- 1c: 讀 DGA 11 檔(family = 檔名去副檔名)----
    dga_rows = []
    for path in sorted(glob.glob(os.path.join(args.dga_dir, "*.txt"))):
        fam = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                d = line.strip()
                if d:
                    dga_rows.append((d, 1, fam))
    dga = pd.DataFrame(dga_rows, columns=["domain", "label", "family"])
    print(f"DGA 讀入 {len(dga)} 筆,家族:{sorted(dga.family.unique())}")

    # (坑一)明確選欄 + group_keys=False,否則 family 欄會消失
    dga_s = (dga.groupby("family", group_keys=False)[["domain", "label", "family"]]
                .apply(lambda g: g.sample(min(len(g), args.family_cap),
                                          random_state=args.seed)))
    print(f"DGA 取樣後 {len(dga_s)} 筆(每家族上限 {args.family_cap})")

    # ---- benign: Tranco(rank,domain,無標題)----
    tr = pd.read_csv(args.tranco, names=["rank", "domain"])
    tr = tr.sample(min(len(tr), args.benign_n), random_state=args.seed)
    benign = pd.DataFrame({"domain": tr["domain"].astype(str).values,
                           "label": 0, "family": "benign"})
    print(f"benign 取樣 {len(benign)} 筆")

    df = pd.concat([dga_s[["domain", "label", "family"]], benign],
                   ignore_index=True)

    # ---- 2: 取 SLD ----
    ext = get_extractor()
    df["sld"] = df["domain"].map(lambda d: ext(str(d)).domain.lower())

    # 濾長度<min_len
    before = len(df)
    df = df[df["sld"].str.len() >= args.min_len].copy()
    print(f"濾長度<{args.min_len}: {before} -> {len(df)}"
          f"(benign 損失 {(1 - (df.label==0).sum()/ (benign.__len__())):.1%} 內含)")

    # (坑二自檢)每家族 SLD 唯一值數應接近筆數;若≈1 表示亂數藏子網域,警告
    for fam in ALL_FAMILIES:
        sub = df[df.family == fam]
        if len(sub) and sub["sld"].nunique() <= max(2, len(sub) * 0.01):
            print(f"  [坑二警告] 家族 {fam} SLD 幾乎全同(nunique={sub['sld'].nunique()}),"
                  f"疑似亂數藏子網域,請檢查!")

    # ---- 合併後去重(防資料洩漏),再切分 ----
    df = df.drop_duplicates(subset="sld").reset_index(drop=True)
    print(f"去重後 {len(df)}(正常 {(df.label==0).sum()} / 惡意 {(df.label==1).sum()})")

    # (坑一 assert)DGA 樣本 family 不可為空
    assert df.loc[df.label == 1, "family"].notna().all(), "family 欄遺失!(踩到坑一)"

    # stratified split,保留 family
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=args.seed)
    df.loc[train_df.index, "split"] = "train"
    df.loc[test_df.index, "split"] = "test"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df[["sld", "label", "family", "split"]].to_csv(args.out, index=False)
    print(f"寫出 {args.out}  train={len(train_df)} test={len(test_df)}")

    # 印出各家族筆數(供 LOO)
    fam_counts = df[df.label == 1].groupby("family").size().to_dict()
    print("各家族(去重後)筆數:", fam_counts)


if __name__ == "__main__":
    main()
