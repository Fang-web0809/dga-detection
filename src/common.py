"""共用工具:字元集/編碼、RF 手工特徵、資料載入。所有 script 共用,確保一致性。"""
import math
import numpy as np
import pandas as pd

# --- 字元集(交接文件第 5 節,index 0 保留給 padding)---
CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789-_"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}  # 1..38
VOCAB_SIZE = len(CHARSET) + 1                          # +1 for padding(0)
MAXLEN = 40

# leave-one-family-out 用的家族分型(交接文件第 5 節)
DICT_FAMILIES = ["matsnu", "suppobox_1", "pizd", "gozi_gpl", "rovnix"]      # 字典/組合型(難)
RANDOM_FAMILIES = ["cryptolocker", "ramnit", "necurs", "tinba", "banjori", "qadars"]  # 亂數型(對照)
ALL_FAMILIES = DICT_FAMILIES + RANDOM_FAMILIES


def encode_domains(domains, maxlen=MAXLEN):
    """把字串序列編碼成 (N, maxlen) int 矩陣;字元集外字元記為 0(同 padding)。"""
    n = len(domains)
    X = np.zeros((n, maxlen), dtype=np.int64)
    for i, d in enumerate(domains):
        d = str(d)[:maxlen]
        for j, ch in enumerate(d):
            X[i, j] = CHAR2IDX.get(ch, 0)
    return X


# --- Random Forest 手工特徵(交接文件第 7 節:自算,不用內建特徵檔)---
_VOWELS = set("aeiou")

def _entropy(s):
    if not s:
        return 0.0
    from collections import Counter
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())

def rf_features(domains):
    """回傳 (N,4) 特徵:length / entropy / digit_ratio / vowel_ratio。"""
    feats = np.zeros((len(domains), 4), dtype=np.float64)
    for i, d in enumerate(domains):
        d = str(d)
        n = len(d)
        if n == 0:
            continue
        digits = sum(ch.isdigit() for ch in d)
        vowels = sum(ch in _VOWELS for ch in d)
        feats[i, 0] = n
        feats[i, 1] = _entropy(d)
        feats[i, 2] = digits / n
        feats[i, 3] = vowels / n
    return feats

RF_FEATURE_NAMES = ["length", "entropy", "digit_ratio", "vowel_ratio"]


def load_processed(path):
    """讀 preprocess 產出的 processed.csv(欄位 sld,label,family,split)。"""
    df = pd.read_csv(path, keep_default_na=False, na_values=[])
    df["label"] = df["label"].astype(int)
    return df
