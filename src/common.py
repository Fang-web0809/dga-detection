"""Shared helpers: character vocabulary/encoding, hand-crafted RF features, data loading."""
import math
import numpy as np
import pandas as pd

# Character vocabulary. Index 0 is reserved for padding.
CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789-_"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}  # 1..38
VOCAB_SIZE = len(CHARSET) + 1                          # +1 for the padding slot
MAXLEN = 40

# Family groups for the leave-one-family-out experiment.
DICT_FAMILIES = ["matsnu", "suppobox_1", "pizd", "gozi_gpl", "rovnix"]      # dictionary / word-based (hard)
RANDOM_FAMILIES = ["cryptolocker", "ramnit", "necurs", "tinba", "banjori", "qadars"]  # arithmetic / random (control)
ALL_FAMILIES = DICT_FAMILIES + RANDOM_FAMILIES


def encode_domains(domains, maxlen=MAXLEN):
    """Encode strings into an (N, maxlen) int matrix; out-of-vocab chars map to 0 (padding)."""
    n = len(domains)
    X = np.zeros((n, maxlen), dtype=np.int64)
    for i, d in enumerate(domains):
        d = str(d)[:maxlen]
        for j, ch in enumerate(d):
            X[i, j] = CHAR2IDX.get(ch, 0)
    return X


# Hand-crafted features for the Random Forest baseline (computed here, not taken
# from any bundled feature file).
_VOWELS = set("aeiou")

def _entropy(s):
    if not s:
        return 0.0
    from collections import Counter
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())

def rf_features(domains):
    """Return an (N, 4) matrix: length, entropy, digit_ratio, vowel_ratio."""
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
    """Load the processed.csv written by preprocess (columns: sld, label, family, split)."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[])
    df["label"] = df["label"].astype(int)
    return df
