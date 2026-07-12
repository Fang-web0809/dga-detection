"""Unit tests for the data-processing helpers and the two documented pitfalls.

Kept dependency-light (numpy + pandas only) so CI runs fast without torch.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import encode_domains, rf_features, CHAR2IDX, MAXLEN, VOCAB_SIZE


def test_encode_shape_and_padding():
    X = encode_domains(["ab", "abcdef"])
    assert X.shape == (2, MAXLEN)
    assert X[0, 0] == CHAR2IDX["a"] and X[0, 1] == CHAR2IDX["b"]
    assert X[0, 2] == 0  # padded


def test_encode_out_of_vocab_and_truncation():
    # '.' and uppercase are out of vocab -> 0; length is capped at MAXLEN.
    X = encode_domains(["A.b"])
    assert X[0, 0] == 0 and X[0, 1] == 0 and X[0, 2] == CHAR2IDX["b"]
    long = "a" * (MAXLEN + 10)
    assert encode_domains([long]).shape[1] == MAXLEN
    assert int(X.max()) < VOCAB_SIZE


def test_rf_features_values():
    F = rf_features(["aa11"])
    length, entropy, digit_ratio, vowel_ratio = F[0]
    assert length == 4
    assert digit_ratio == 0.5      # 2 of 4 are digits
    assert vowel_ratio == 0.5      # 2 of 4 are vowels ('a')
    assert entropy > 0


def test_family_label_survives_groupby_sample():
    """Pitfall #1: group_keys=False + explicit column selection must keep `family`."""
    df = pd.DataFrame({
        "domain": [f"d{i}" for i in range(20)],
        "label": [1] * 20,
        "family": ["a"] * 10 + ["b"] * 10,
    })
    sampled = (df.groupby("family", group_keys=False)[["domain", "label", "family"]]
                 .apply(lambda g: g.sample(3, random_state=42)))
    assert "family" in sampled.columns
    assert sampled["family"].notna().all()
    assert set(sampled["family"].unique()) == {"a", "b"}


def test_dedup_prevents_leakage():
    """Deduplicating on sld before splitting removes cross-set duplicates."""
    df = pd.DataFrame({"sld": ["x", "x", "y", "z"], "label": [1, 0, 1, 0]})
    assert len(df.drop_duplicates(subset="sld")) == 3
