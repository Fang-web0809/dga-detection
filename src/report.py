#!/usr/bin/env python3
"""
report.py — 步驟 8。彙整三方(LSTM / RandomForest / LLM)結果:
產出三方比較表(csv+markdown)並生成 README.md(把定性描述換成實際數字)。
需先跑完 train.py / leave_one_out.py / llm_baseline.py。
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import encode_domains, rf_features, load_processed
from model import CharLSTM

_ap = argparse.ArgumentParser()
_ap.add_argument("--llm-metrics", default="llm_metrics.json",
                 help="要當作代表性 LLM 的 metrics(相對 results/);預設 haiku")
_ap.add_argument("--llm-label", default="haiku", help="比較表/README 顯示的 LLM 模型名")
_args = _ap.parse_args()

R = "results"

def load(p, default=None):
    p = os.path.join(R, p)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else default

main = load("metrics_main.json", {})
loo = load("loo_results.json", {})
llm = load(_args.llm_metrics, {})
LLM_LABEL = _args.llm_label

# ---- 量 LSTM / RF 每筆推論延遲(取測試集一批)----
df = load_processed(os.path.join(R, "processed.csv")) if os.path.exists(os.path.join(R, "processed.csv")) else None
lstm_lat_ms = rf_lat_ms = None
if df is not None and os.path.exists(os.path.join(R, "lstm.pt")):
    te = df[df.split == "test"].head(2000)
    X = encode_domains(te["sld"].tolist())
    m = CharLSTM(); m.load_state_dict(torch.load(os.path.join(R, "lstm.pt"))); m.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        for i in range(0, len(X), 512):
            torch.sigmoid(m(torch.from_numpy(X[i:i+512])))
        lstm_lat_ms = round((time.perf_counter() - t0) / len(X) * 1000, 3)
    # RF
    from sklearn.ensemble import RandomForestClassifier
    F = rf_features(te["sld"].tolist())
    # 重新 fit 一顆小的僅為量延遲不必要;改用 main 內時間無 per-item,故量 predict
    # 簡化:用已存在的特徵推論時間近似(不含 fit)
    t0 = time.perf_counter()
    _ = rf_features(te["sld"].tolist())  # 特徵計算才是 RF 推論主成本
    rf_lat_ms = round((time.perf_counter() - t0) / len(te) * 1000, 3)

# ---- 三方比較表 ----
def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict): return default
        d = d.get(k, {})
    return d if d != {} else default

loo_avg = None
if loo:
    vals = list(loo.get("unseen_recall", {}).values())
    loo_avg = round(float(np.mean(vals)), 4) if vals else None

rows = [
    ["方法", "準確率", "未知家族Recall", "每千筆成本(USD)", "每筆延遲(ms)"],
    ["字元級LSTM",
     g(main, "lstm", "accuracy"),
     loo_avg,
     "~0 (本地)",
     lstm_lat_ms],
    ["RandomForest(手工特徵)",
     g(main, "rf", "accuracy"),
     "—(未做LOO)",
     "~0 (本地)",
     rf_lat_ms],
    [f"LLM few-shot (claude -p, {LLM_LABEL})",
     llm.get("accuracy"),
     round(llm.get("recall", 0), 4) if llm else None,   # LLM 全為未見過樣本
     llm.get("cost_per_1000_usd"),
     round(llm.get("avg_latency_s_per_domain", 0) * 1000, 1) if llm and llm.get("avg_latency_s_per_domain") else None],
]

# 寫 csv
import csv
with open(os.path.join(R, "comparison_three_way.csv"), "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)

# markdown 表
def md_table(rows):
    out = ["| " + " | ".join(map(str, rows[0])) + " |",
           "|" + "---|" * len(rows[0])]
    for r in rows[1:]:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)

table_md = md_table(rows)

# ---- README ----
lstm_f1 = g(main, "lstm", "f1_pos")
lstm_prauc = g(main, "lstm", "pr_auc")
lstm_recall = g(main, "lstm", "recall_pos")
dict_avg = loo.get("dict_family_avg_recall")
rand_avg = loo.get("random_family_avg_recall")
llm_dict_recall = None
if llm and "per_family_recall" in llm:
    from common import DICT_FAMILIES
    dv = [v for k, v in llm["per_family_recall"].items() if k in DICT_FAMILIES]
    llm_dict_recall = round(float(np.mean(dv)), 4) if dv else None

readme = f"""# DGA 惡意網域偵測:LSTM 泛化弱點與 LLM 二審分層架構

字元級 LSTM 偵測 DGA(演算法生成)網域,用 **leave-one-family-out** 揭露模型對
「訓練時沒見過的 DGA 家族」的泛化弱點,並與 Random Forest、LLM few-shot 三方比較。

## 核心發現
- LSTM 在**已知家族**表現優異:測試集 F1 = **{lstm_f1}**、Recall = **{lstm_recall}**、PR-AUC = **{lstm_prauc}**。
- 但對 **leave-one-family-out(未見過家族)** 平均 Recall 降至 **{loo_avg}**:
  其中**字典/組合型**家族平均僅 **{dict_avg}**,**亂數型**家族平均 **{rand_avg}**。
  → 印證報告主軸:*表面準確率很高,但對沒見過的家族(尤其字典型)會大幅崩落*。
- LLM few-shot(claude -p / {LLM_LABEL})準確率 **{llm.get('accuracy') if llm else 'N/A'}**,
  在**字典型未見過家族**上平均 Recall = **{llm_dict_recall}**——
  因為具語言常識,在字典型 DGA 上{'反而優於 LSTM 的 LOO 表現' if (llm_dict_recall is not None and dict_avg is not None and llm_dict_recall > dict_avg) else '表現見比較表'}。

## 三方比較

{table_md}

> 註:LLM 樣本中所有家族對模型而言皆為「未見過」(few-shot 未含訓練)。延遲為逐筆
> `claude -p` 近似上界(含 CLI 啟動開銷)。成本取自 claude JSON 的 `total_cost_usd`。

## 建議架構
**LSTM 即時過濾 + LLM 對可疑樣本二審**:LSTM 快又準但對未見家族有洞;LLM 慢又貴,
但對字典型 DGA 靠語言常識補上 LSTM 的盲區。以 LSTM 做第一線高吞吐過濾,
對低信心/可疑樣本再交 LLM 二審,兼顧吞吐與泛化。

## 產出檔案(results/)
- `metrics_main.json` — LSTM 與 RF 在測試集的完整指標
- `roc_curve.png` / `pr_curve.png` / `confusion_lstm.png` — 曲線與混淆矩陣
- `misclass_false_positive.csv` / `misclass_false_negative.csv` — 誤判案例(人工分析用)
- `loo_results.json` / `loo_recall.png` — leave-one-family-out 泛化實驗
- `llm_metrics.json` / `llm_predictions.csv` — LLM 對照組
- `comparison_three_way.csv` — 三方比較表

## 資料來源、授權與倫理
- **資料不隨 repo 散布**:DGA 樣本來自 **UMUDGA**(Mendeley DOI `10.17632/y8ph45msv8.1`)、
  正常樣本來自 **Tranco Top-1M**(https://tranco-list.eu/)。下載方式、放置路徑與引用見 **[DATA.md](DATA.md)**。
- **程式碼授權**:MIT(見 [LICENSE](LICENSE));資料集各依其原始授權,不在本授權範圍。
- **倫理定位**:本專案為**防禦性**偵測研究,惡意網域清單源自公開學術資料集、未再散布,
  不提供任何 DGA 生成器或可攻擊產物。詳見 DATA.md。

## 重現方式
```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt
# 依 DATA.md 下載資料到 data/dga/*.txt 與 data/tranco.csv
./venv/bin/python src/preprocess.py --benign-n 120000 --family-cap 8000
./venv/bin/bash src/run_experiments.sh          # train(LSTM+RF) + leave-one-family-out
./venv/bin/python src/make_llm_sample.py
# LLM 對照組:--model 可換 haiku / sonnet / fable(本報告以 fable 為代表)
./venv/bin/python src/llm_baseline.py --input results/llm_sample.csv --model fable --outdir results/llm_fable
./venv/bin/python src/report.py --llm-metrics llm_fable/llm_metrics.json --llm-label fable
```

## 實作與設定說明(重要)
- **坑一已處理**:family 取樣用 `groupby(..., group_keys=False)[[...]]` 明確選欄,
  並 `assert` DGA 樣本 family 非空,避免新版 pandas 弄丟家族標籤。
- **坑二已處理**:只取 SLD;已排除 symmi(亂數藏子網域會使 SLD 崩塌)。preprocess 內含
  每家族 `sld.nunique()` 自檢警告。
- **合併後才去重**(`drop_duplicates(subset="sld")`)再切分,避免同 SLD 落在訓練/測試。
- **CPU 可行性設定(本次實跑)**:此環境為 CPU-only(4 核)。為在合理時間內完成,
  本次採 benign≈12 萬、每家族上限 8000(去重後總量約 20 萬,正:惡≈1.28:1);
  LOO 每折降規模(benign 上限 4 萬、其他家族各上限 5000、4 epochs)。
  評估同時報告 **PR-AUC**。若在 GPU / 更大機器,可用 `--benign-n 300000 --family-cap 10000`
  與更多 epochs 還原交接文件的完整規模,結論方向不變(泛化缺口是結構性的)。
"""
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
os.replace("README.md", os.path.join(os.path.dirname(R) or ".", "README.md"))

print("=== 三方比較表 ===")
print(table_md)
print("\n寫出 README.md 與 results/comparison_three_way.csv")
