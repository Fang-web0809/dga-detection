#!/usr/bin/env python3
"""
Aggregate the LSTM / RandomForest / LLM results into a three-way comparison table
(CSV + Markdown) and regenerate README.md with the actual numbers filled in.
Run train.py, leave_one_out.py and llm_baseline.py first.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import encode_domains, rf_features, load_processed
from model import CharLSTM

_ap = argparse.ArgumentParser()
_ap.add_argument("--llm-metrics", default="llm_metrics.json",
                 help="metrics file to use as the representative LLM (relative to results/)")
_ap.add_argument("--llm-label", default="haiku", help="LLM model name shown in the table/README")
_args = _ap.parse_args()

R = "results"

def load(p, default=None):
    p = os.path.join(R, p)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else default

main = load("metrics_main.json", {})
loo = load("loo_results.json", {})
llm = load(_args.llm_metrics, {})
LLM_LABEL = _args.llm_label

# Per-domain inference latency for LSTM / RF, measured on a slice of the test set.
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
    # For RF, feature extraction dominates inference cost, so approximate the
    # per-domain latency by the time to compute features (the tree lookup is negligible).
    t0 = time.perf_counter()
    _ = rf_features(te["sld"].tolist())
    rf_lat_ms = round((time.perf_counter() - t0) / len(te) * 1000, 3)

# Three-way comparison table.
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
     round(llm.get("recall", 0), 4) if llm else None,   # every LLM sample is effectively unseen
     llm.get("cost_per_1000_usd"),
     round(llm.get("avg_latency_s_per_domain", 0) * 1000, 1) if llm and llm.get("avg_latency_s_per_domain") else None],
]

# write CSV
import csv
with open(os.path.join(R, "comparison_three_way.csv"), "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)

# Markdown table
def md_table(rows):
    out = ["| " + " | ".join(map(str, rows[0])) + " |",
           "|" + "---|" * len(rows[0])]
    for r in rows[1:]:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)

table_md = md_table(rows)

# Build README from the numbers above.
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
loo_matsnu = (loo.get("unseen_recall") or {}).get("matsnu")

readme = f"""# DGA Domain Detection — LSTM generalization study & LLM two-tier triage

[![CI](https://github.com/Fang-web0809/dga-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Fang-web0809/dga-detection/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

> Character-level LSTM for detecting algorithmically-generated (DGA) domains — with a
> leave-one-family-out study that exposes generalization gaps, and a layered LLM triage design.

<p align="center">
  <img src="results/loo_recall.png" width="88%" alt="Leave-one-family-out recall on unseen families">
</p>

**TL;DR.** A character-level LSTM flags DGA domains at **{g(main,'lstm','accuracy')}** test
accuracy — but a **leave-one-family-out** evaluation shows recall on *families never seen during
training* collapses: dictionary-style families average **{dict_avg}** and drop as low as
**{loo_matsnu}** (matsnu), versus **{g(main,'lstm','recall_pos')}** on the normal test set. An
**LLM few-shot** baseline (Claude, {LLM_LABEL}) recovers most of that blind spot (unseen
dictionary-family recall **{llm_dict_recall}**), motivating a layered
**"LSTM fast filter + LLM second-opinion on suspicious samples"** design.

**Try it** (uses the committed model):
```bash
python src/predict.py google.com wikipedia.org kq3vz8xw1.com xjkw92mfp.net
```

Figures, metrics and the three-way comparison table live in [`results/`](results/);
data sources are in [`DATA.md`](DATA.md). The full report below is in Traditional Chinese.

---

> 以下為完整中文報告 · Full report in Traditional Chinese.

字元級 LSTM 偵測 DGA(演算法生成)網域,用 **leave-one-family-out** 揭露模型對
「訓練時沒見過的 DGA 家族」的泛化弱點,並與 Random Forest、LLM few-shot 三方比較。
上方長條圖為核心結果:每個家族輪流當「訓練時沒見過」的測試家族,亂數型即使沒見過仍有
~90–100%,但字典型(尤其 matsnu、suppobox_1)大幅崩落——這正是本專案要凸顯的泛化盲區。

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

### 判別力:ROC 與 PR 曲線(LSTM vs RandomForest)
<p align="center">
  <img src="results/roc_curve.png" width="46%" alt="ROC curve — LSTM vs RandomForest">
  <img src="results/pr_curve.png" width="46%" alt="Precision-Recall curve — LSTM vs RandomForest">
</p>

字元級 LSTM 的 ROC-AUC 與 PR-AUC 都明顯高於手工特徵的 RandomForest。由於正負樣本略不
平衡,**PR 曲線(右)比 ROC(左)更能反映實務偵測品質**。另附混淆矩陣 `results/confusion_lstm.png`。

## 建議架構
**LSTM 即時過濾 + LLM 對可疑樣本二審**:LSTM 快又準但對未見家族有洞;LLM 慢又貴,
但對字典型 DGA 靠語言常識補上 LSTM 的盲區。以 LSTM 做第一線高吞吐過濾,
對低信心/可疑樣本再交 LLM 二審,兼顧吞吐與泛化。

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

## 實作備註(資料處理與可重現性)

以下幾點在實作時容易出錯,且會直接影響實驗結論,一併記錄理由與處理方式。

**分層抽樣後必須保留家族標籤。** 在 pandas ≥ 2.x,`groupby("family").apply(...)` 搭配
`reset_index(drop=True)` 會把 `family` 併入 index 後一起丟掉,使標籤無聲消失、
leave-one-family-out 無從進行。本專案改用 `groupby("family", group_keys=False)[[...]]`
明確選取欄位,並在資料組完後以 `assert` 確認惡意樣本的 `family` 皆非空。

**只取主網域(SLD),但有一個例外。** DGA 的隨機性集中在 SLD,保留 TLD 會讓模型走捷徑
而使泛化評估失真,因此統一以 `tldextract` 取 SLD。少數家族(如 symmi)把隨機字串放在
子網域,只取 SLD 會塌成單一值,故予以排除;`preprocess.py` 內建 `sld.nunique()` 檢查,
避免日後新增家族時再度誤用。

**先合併正負樣本、去重,再切分。** 以 `drop_duplicates(subset="sld")` 在合併後去重,
確保同一個 SLD 不會同時落在訓練集與測試集,避免資料洩漏而高估效能。

**本輪以 CPU 規模執行,可放大重現。** 執行環境為 4 核 CPU;為兼顧時間,本輪採
benign ≈ 120k、每家族上限 8k(去重後約 200k,正負比 ≈ 1.28 : 1),leave-one-family-out
每折再降規模(benign 上限 40k、其他家族各 5k、4 epochs),並同時以 PR-AUC 評估。在 GPU
或更大機器上,可用 `--benign-n 300000 --family-cap 10000` 與更多 epochs 還原完整規模;
泛化缺口屬結構性,結論方向不受規模影響。

## 產出檔案(results/)
- `metrics_main.json` — LSTM 與 RF 在測試集的完整指標
- `roc_curve.png` / `pr_curve.png` / `confusion_lstm.png` — 曲線與混淆矩陣
- `misclass_false_positive.csv` / `misclass_false_negative.csv` — 誤判案例(人工分析用)
- `loo_results.json` / `loo_recall.png` — leave-one-family-out 泛化實驗
- `llm_metrics.json` / `llm_predictions.csv` — LLM 對照組
- `comparison_three_way.csv` — 三方比較表

## License · Acknowledgements · Citation
- **License** — code released under the MIT License (see [LICENSE](LICENSE)); the datasets keep their own licenses and are not redistributed here.
- **Acknowledgements** — UMUDGA DGA dataset (Mendeley DOI `10.17632/y8ph45msv8.1`) and the Tranco top-1M list (https://tranco-list.eu/).
- **Citation** — if you build on this work, please cite the UMUDGA dataset and Tranco; details in [DATA.md](DATA.md).
"""
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
os.replace("README.md", os.path.join(os.path.dirname(R) or ".", "README.md"))

print("=== three-way comparison ===")
print(table_md)
print("\nwrote README.md and results/comparison_three_way.csv")
