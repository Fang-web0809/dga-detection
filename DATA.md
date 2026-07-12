# 資料來源、取得方式與授權

本專案**不散布任何原始資料集**(`data/` 已列入 `.gitignore`)。請依下列說明自行下載,
放到對應路徑後即可重現整條 pipeline。

## 需要的資料

| 用途 | 來源 | 放置路徑 |
|---|---|---|
| 惡意樣本(DGA,label=1) | UMUDGA 資料集 | `data/dga/<family>.txt`(11 個家族,一行一個 FQDN) |
| 正常樣本(benign,label=0) | Tranco Top-1M | `data/tranco.csv`(兩欄 `rank,domain`,無標題) |

使用的 11 個家族:`banjori, cryptolocker, gozi_gpl, matsnu, necurs, pizd, qadars,
ramnit, rovnix, suppobox_1, tinba`(family 標籤 = 檔名去副檔名)。

> 註:已刻意**排除 symmi 家族**。其亂數藏在子網域,只取 SLD 會使整個家族塌成單一值,
> 不適用本專案「只取主網域」的設定。

## 下載與引用

- **UMUDGA**(University of Murcia DGA dataset)
  - DOI: `10.17632/y8ph45msv8.1`(Mendeley Data)
  - 下載後把各家族的原始網域清單放進 `data/dga/`,檔名即家族名。
  - 請依 Mendeley Data 頁面標示的授權條款(通常為 CC BY 4.0)使用並**附上引用**。

- **Tranco Top-1M**
  - 網站: https://tranco-list.eu/
  - 下載任一份 Top-1M 清單,解壓後將 `top-1m.csv` 放為 `data/tranco.csv`。
  - Tranco 為可重現的網域排名研究清單,使用時請引用其論文/清單 ID。

## 倫理與使用聲明

本專案為**防禦性資安研究**:目的是評估 DGA 偵測模型對「未見過家族」的泛化弱點,
並探討「LSTM 即時過濾 + LLM 二審」的分層偵測架構。

- 惡意網域清單來自**公開的學術研究資料集**,本 repo **未再散布**這些清單,
  也不提供任何 DGA 生成器或可直接用於攻擊的產物。
- `results/` 內作為誤判分析範例的少量 SLD 片段,屬資料集之衍生樣本,僅供研究說明之用。
- 請勿將本專案用於產生、散布或濫用惡意網域。
