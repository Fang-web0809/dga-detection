#!/usr/bin/env python3
"""
Classify domains as DGA or benign with the trained character-level LSTM.

Uses the committed model at results/lstm.pt, so it works right after cloning.
Each input is reduced to its SLD (the representation the model was trained on).

Usage:
    python src/predict.py google.com kq3vz8xw1.com
    echo "somedomain.net" | python src/predict.py
"""
import os
import sys

import torch
import tldextract

sys.path.insert(0, os.path.dirname(__file__))
from common import encode_domains
from model import CharLSTM

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "lstm.pt")


def load_model(path=MODEL_PATH):
    model = CharLSTM()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def predict(domains, model=None, ext=None):
    """Return a list of (domain, sld, p_dga) for each input domain."""
    model = model or load_model()
    ext = ext or tldextract.TLDExtract()
    slds = [ext(str(d)).domain.lower() or str(d).lower() for d in domains]
    X = encode_domains(slds)
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X))).numpy()
    return list(zip(domains, slds, probs))


def main():
    args = sys.argv[1:]
    if not args:  # fall back to stdin
        args = [line.strip() for line in sys.stdin if line.strip()]
    if not args:
        sys.exit("usage: python src/predict.py <domain> [domain ...]")
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"model not found: {MODEL_PATH}. Train it first (see README).")

    for dom, sld, prob in predict(args):
        verdict = "DGA" if prob >= 0.5 else "benign"
        print(f"{dom:<28} sld={sld:<20} p(DGA)={prob:.3f}  -> {verdict}")


if __name__ == "__main__":
    main()
