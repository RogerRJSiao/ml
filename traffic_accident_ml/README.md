# 交通事故嚴重程度預測與高風險情境分析 🚗🛞

透過台灣歷年交通事故的公開資料，可用資料欄位包括天候、光線、道路型態、號誌、肇因、當事者屬性、經緯度等，實作歸檔管理、機器學習、可視化 pipeline 功能。

## 原始資料來源 💾

| 資料集名稱 | 檔案類型 | 初次上架日 | 備註 |
| --- | --- | --- | --- |
| [109年傷亡道路交通事故資料](https://data.gov.tw/dataset/158864) | 單一zip打包多個csv | 2022-10-13 | |
| [110年傷亡道路交通事故資料](https://data.gov.tw/dataset/158865) | 單一zip打包多個csv | 2022-10-13 | |
| [111年傷亡道路交通事故資料](https://data.gov.tw/dataset/161199) | 單一zip打包多個csv | 2023-03-08 | |
| [112年傷亡道路交通事故資料](https://data.gov.tw/dataset/167905) | 單一zip打包多個csv | 2024-03-04 | |
| [113年傷亡道路交通事故資料](https://data.gov.tw/dataset/172969) | 單一zip打包多個csv | 2025-03-03 | |
| [114年傷亡道路交通事故資料](https://data.gov.tw/dataset/177136) | 單一zip打包多個csv | 2026-03-06 | |
| [即時交通事故資料 (A1類)(json格式)](https://data.gov.tw/dataset/57023) | json | 2017-10-17 | 持續更新，僅列當年度 |
| [即時交通事故資料 (A2類)(json格式)](https://data.gov.tw/dataset/57024) | 單一zip打包單一json | 2017-10-17 | 持續更新，僅列當年度 |

### A1 類/ A2 類交通事故的資料欄位 (50 欄)

**時間資訊**
發生年度、發生月份、發生日期、發生時間

**事故基本資訊**
事故類別名稱、處理單位名稱警局層、發生地點、事故類型及型態大類別名稱、事故類型及型態子類別名稱、肇事逃逸類別名稱-是否肇逃

**環境條件**
天候名稱、光線名稱

**道路與交通設施**
道路類別-第1當事者-名稱、速限-第1當事者、道路型態大類別名稱、道路型態子類別名稱、事故位置大類別名稱、事故位置子類別名稱、路面狀況-路面鋪裝名稱、路面狀況-路面狀態名稱、路面狀況-路面缺陷名稱、道路障礙-障礙物名稱、道路障礙-視距品質名稱、道路障礙-視距名稱、號誌-號誌種類名稱、號誌-號誌動作名稱、車道劃分設施-分向設施大類別名稱、車道劃分設施-分向設施子類別名稱、車道劃分設施-分道設施-快車道或一般車道間名稱、車道劃分設施-分道設施-快慢車道間名稱、車道劃分設施-分道設施-路面邊線名稱

**肇因研判**
肇因研判大類別名稱-主要、肇因研判子類別名稱-主要、肇因研判大類別名稱-個別、肇因研判子類別名稱-個別

**當事者屬性與行為**
當事者順位、當事者區分-類別-大類別名稱-車種、當事者區分-類別-子類別名稱-車種、當事者屬-性-別名稱、當事者事故發生時年齡、保護裝備名稱、行動電話或電腦或其他相類功能裝置名稱、當事者行動狀態大類別名稱、當事者行動狀態子類別名稱、車輛撞擊部位大類別名稱-最初、車輛撞擊部位子類別名稱-最初、車輛撞擊部位大類別名稱-其他、車輛撞擊部位子類別名稱-其他

**傷亡結果與地理位置**
死亡受傷人數、經度、緯度

> 台灣交通事故等級判定
> - A1 類：造成人員當場或 24 小時內死亡之交通事故。
> - A2 類：造成人員受傷或超過 24 小時死亡之交通事故。
> - A3 類：指僅有車輛財物受損之交通事故。(由於資料不完整，故本專案暫不列入)

## 資料夾架構 📂

```
traffic_accident_ml/
├── data/
│   ├── incoming/               # 下載、待解壓縮的原始 zip
│   ├── raw/Y<yyyy>/            # extractor.py 解壓縮出的逐年 csv 原始檔
│   └── processed/
│       ├── merged_years/       # merger.py 產出的年度匯總檔
│       └── cleaned/            # 未來欄位層級跨年度清洗完成的資料集
├── registry/
│   ├── extracted_zips.json     # extractor.py 記錄已解壓縮 zip 的 sha256，避免重複解壓
│   ├── merged_years.json       # merger.py 記錄各年度來源檔案的 sha256，避免重複合併
│   └── history/                # 每次執行 check_update.py 的比對紀錄
├── models/                     # 訓練完成的模型檔與 metadata.json
├── requirements.txt
└── src/
    ├── schema.py                    # 固定欄位 schema 定義（50+ 欄位），供 loader.py 驗證用
    ├── data_management/
    │   ├── common.py                # 共用工具
    │   ├── extractor.py             # 解壓縮年別 zip
    │   ├── merger.py                # 合併年別 csv
    │   └── check_update.py          # 彙整匯入 pipeline 各年度完成到哪個階段的報表
    ├── preprocessing/
    │   └── loader.py                # 讀取多年度 csv、依 schema 驗證，合併成單一 DataFrame
    ├── features/
    │   └── pca.py                   # sklearn PCA 封裝：fit / transform / 匯出
    ├── training/
    │   └── train.py                 # loader -> 前處理 -> (PCA) -> 監督式學習模型 -> 匯出至 models/
    ├── visualization/
    │   └── plots.py                 # 可視化 pipeline：資料分布圖、PCA 投影圖、特徵重要性等
    └── predictor.py                 # 對外唯一入口，供其他 py 檔案 import 使用訓練完成的模型
```

## 模型開發與部署規畫 (TBC)

> 以下各階段皆為手動觸發，沒有排程或自動化機制；每個階段都需要人工確認後才執行下一步。

### 訓練階段
1. 將各年度原始資料 zip 放入 `data/incoming/`。
2. 手動執行 `python -m src.data_management.extractor` 依年度解壓縮到 `data/raw/Y<年度>/`。
3. 手動執行 `python -m src.data_management.merger` 將各年度 csv 合併到 `data/processed/merged_years/`。
4. 手動執行 `python -m src.data_management.check_update` 檢查資料是否有更新、是否建議重新訓練（僅回報，不自動觸發）。
5. 人工判讀第 4 步的報表後，若確認需要重新訓練，手動執行 `python -m src.training.train`：內部依序跑 loader → 前處理 → (PCA) → 監督式學習模型，並將模型檔與 `metadata.json` 輸出至 `models/`。

### 評估階段
6. `train.py` 於保留的測試集上計算評估指標（如準確率、F1-score、混淆矩陣），連同訓練資料版本、使用的特徵一併寫入 `models/metadata.json`，作為不同版本模型的比較依據。
7. 手動執行 `python -m src.visualization.plots` 產出對應圖表（如混淆矩陣熱圖、PCA 投影圖、特徵重要性圖），輔助人工判斷這次訓練出的模型是否可以取代目前使用中的版本。
8. 人工比對新舊模型的評估指標與圖表後，決定是否手動將新模型標記為採用版本（例如更新 `metadata.json` 中的 active 版本欄位）。

### 推論階段
9. 其他 py 檔案透過以下方式複用目前採用中的模型：
   ```python
   from traffic_accident_ml.src.predictor import load_model, predict

   model = load_model()
   result = predict(model, df)
   ```
