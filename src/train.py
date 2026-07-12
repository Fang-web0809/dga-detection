#!/usr/bin/env python3
"""
Train the character-level LSTM and a Random Forest baseline, then evaluate on the
test set: classification report, ROC-AUC + curve, confusion matrix, FPR, PR-AUC,
and a sample of misclassified cases. Outputs go to results/.

Usage: python src/train.py --data results/processed.csv --epochs 8
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                             confusion_matrix, average_precision_score,
                             precision_recall_curve)

sys.path.insert(0, os.path.dirname(__file__))
from common import encode_domains, rf_features, RF_FEATURE_NAMES, load_processed
from model import CharLSTM


def train_lstm(X_tr, y_tr, epochs=8, batch_size=512, lr=1e-3, patience=2,
               val_frac=0.1, seed=42, verbose=True, log=print):
    """Train CharLSTM with early stopping on a held-out validation slice.

    Returns (model, history). Reused by the leave-one-family-out experiment.
    """
    torch.manual_seed(seed)
    n = len(X_tr)
    idx = np.random.RandomState(seed).permutation(n)
    n_val = int(n * val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def loader(ix, shuffle):
        ds = TensorDataset(torch.from_numpy(X_tr[ix]),
                           torch.from_numpy(y_tr[ix].astype(np.float32)))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    tr_loader = loader(tr_idx, True)
    val_loader = loader(val_idx, False)

    model = CharLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()

    best_val, best_state, wait, hist = float("inf"), None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        for xb, yb in tr_loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                l = crit(model(xb), yb)
                vloss += l.item() * len(yb); vn += len(yb)
        vloss /= max(vn, 1)
        dt = time.perf_counter() - t0
        hist.append({"epoch": ep, "val_loss": round(vloss, 5), "sec": round(dt, 1)})
        if verbose:
            log(f"    epoch {ep}/{epochs}  val_loss={vloss:.5f}  ({dt:.1f}s)")
        if vloss < best_val - 1e-4:
            best_val, best_state, wait = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                if verbose: log(f"    early stop @ epoch {ep}")
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, hist


def lstm_predict_proba(model, X, batch_size=1024):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size])
            outs.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/processed.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)

    df = load_processed(args.data)
    tr = df[df.split == "train"]; te = df[df.split == "test"]
    Xtr = encode_domains(tr["sld"].tolist()); ytr = tr["label"].values
    Xte = encode_domains(te["sld"].tolist()); yte = te["label"].values
    print(f"train={len(tr)} test={len(te)}  pos_rate(train)={ytr.mean():.3f}")

    print("training LSTM ...")
    t0 = time.perf_counter()
    model, hist = train_lstm(Xtr, ytr, epochs=args.epochs, batch_size=args.batch_size)
    lstm_train_sec = time.perf_counter() - t0
    torch.save(model.state_dict(), os.path.join(args.outdir, "lstm.pt"))
    p_lstm = lstm_predict_proba(model, Xte)
    yhat_lstm = (p_lstm >= 0.5).astype(int)

    # Random Forest baseline on the hand-crafted features.
    print("training RandomForest baseline ...")
    Ftr = rf_features(tr["sld"].tolist()); Fte = rf_features(te["sld"].tolist())
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    t0 = time.perf_counter(); rf.fit(Ftr, ytr); rf_train_sec = time.perf_counter() - t0
    p_rf = rf.predict_proba(Fte)[:, 1]; yhat_rf = (p_rf >= 0.5).astype(int)

    def block(name, y, yhat, p):
        rep = classification_report(y, yhat, output_dict=True, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        return {
            "accuracy": round(rep["accuracy"], 4),
            "precision_pos": round(rep["1"]["precision"], 4),
            "recall_pos": round(rep["1"]["recall"], 4),
            "f1_pos": round(rep["1"]["f1-score"], 4),
            "roc_auc": round(roc_auc_score(y, p), 4),
            "pr_auc": round(average_precision_score(y, p), 4),
            "fpr": round(fp / (fp + tn), 4) if (fp + tn) else None,
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }

    metrics = {
        "n_train": len(tr), "n_test": len(te),
        "train_pos_rate": round(float(ytr.mean()), 4),
        "lstm": block("lstm", yte, yhat_lstm, p_lstm),
        "rf": block("rf", yte, yhat_rf, p_rf),
        "lstm_train_sec": round(lstm_train_sec, 1),
        "rf_train_sec": round(rf_train_sec, 1),
        "lstm_history": hist,
        "rf_feature_importance": dict(zip(RF_FEATURE_NAMES,
                                          [round(x, 4) for x in rf.feature_importances_])),
    }
    with open(os.path.join(args.outdir, "metrics_main.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ROC
    plt.figure(figsize=(6, 5))
    for name, p in [("LSTM", p_lstm), ("RandomForest", p_rf)]:
        fpr_c, tpr_c, _ = roc_curve(yte, p)
        plt.plot(fpr_c, tpr_c, label=f"{name} (AUC={roc_auc_score(yte, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC — DGA detection"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "roc_curve.png"), dpi=120); plt.close()

    # Precision-Recall
    plt.figure(figsize=(6, 5))
    for name, p in [("LSTM", p_lstm), ("RandomForest", p_rf)]:
        prec, rec, _ = precision_recall_curve(yte, p)
        plt.plot(rec, prec, label=f"{name} (PR-AUC={average_precision_score(yte, p):.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall — DGA detection"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "pr_curve.png"), dpi=120); plt.close()

    # Confusion matrix (LSTM)
    cm = confusion_matrix(yte, yhat_lstm, labels=[0, 1])
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center",
                 color="white" if v > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["benign", "DGA"]); plt.yticks([0, 1], ["benign", "DGA"])
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title("LSTM confusion matrix")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "confusion_lstm.png"), dpi=120); plt.close()

    # Save a sample of misclassified cases for manual inspection.
    te2 = te.copy(); te2["pred"] = yhat_lstm; te2["proba"] = np.round(p_lstm, 3)
    fp_cases = te2[(te2.label == 0) & (te2.pred == 1)].head(30)  # benign flagged as DGA
    fn_cases = te2[(te2.label == 1) & (te2.pred == 0)].head(30)  # DGA missed
    fp_cases.to_csv(os.path.join(args.outdir, "misclass_false_positive.csv"), index=False)
    fn_cases.to_csv(os.path.join(args.outdir, "misclass_false_negative.csv"), index=False)

    print("\n=== main results ===")
    print(json.dumps({"lstm": metrics["lstm"], "rf": metrics["rf"]},
                     ensure_ascii=False, indent=2))
    print("wrote: metrics_main.json, roc_curve.png, pr_curve.png, confusion_lstm.png, "
          "misclass_*.csv, lstm.pt")


if __name__ == "__main__":
    main()
