# 技術文件：食品標示 OCR 分析系統 (Side Project) 🍔🔍

## 1. 開發目的

本專案目的是建構一個簡單的 OCR 圖片文字辨識服務，專門分析食品包裝或標示，並回傳以下資訊給使用者：

- 成分列表
- 營養成分與熱量計算
- 過敏原判定（比對預設過敏原清單）
- 素食類別判定（全素、蛋奶素、非素食等）
- 標註圖片（可選是否下載）
- 圖片 Metadata 存檔（Excel，用於後續資料追蹤）

**技術目標：**

- 使用 FastAPI 作為 RESTful API 後端
- 提供簡單靜態前端 HTML + Bootstrap 上傳圖片
- 使用 ngrok 提供外網測試連線
- 提供 HTTP Header Token 驗證
- 支援檔案檢查（格式與大小）
- 將後端業務邏輯與 API 層分離，便於維護與測試（學習架構設計）

## 2. 專案檔案結構

```
food_label_ocr/
│
├── main.py                  # FastAPI 入口
├── requirements.txt         # Python 套件列表
├── uploads/                 # 上傳圖片 & 標註圖片儲存
│
├── app/
│   ├── __init__.py
│   │
│   ├── api/                 # API 層，處理 HTTP 請求、Token 驗證
│   │   └── routes.py
│   │
│   ├── services/            # 業務邏輯層
│   │   └── food_processing.py  # OCR、文字解析、熱量/過敏原/素食判定、標註圖片
│   │
│   ├── models/              # 資料結構層
│   │   └── food_models.py       # Ingredient、Nutrition、FoodItem
│   │
│   └── utils/               # 工具層
│       ├── file_utils.py        # 檔案儲存、檢查、安全命名
│       └── metadata_utils.py    # 讀取圖片 Metadata + Excel 存檔
│
└── static/
    └── index.html           # 前端靜態頁面，Bootstrap + JS
```

## 3. 功能細項

| 功能              | 說明                  | 技術實作                             | 風險 / 注意事項             |
|-------------------|-----------------------|--------------------------------------|-----------------------------|
| **圖片上傳**      | 使用者上傳食品圖片   | FastAPI `UploadFile`，檢查檔案類型 & 大小 | 檔案過大可能耗記憶體；檔案非圖片需拒絕 |
| **Token 驗證**    | 保護 API 不被濫用   | HTTP Header `x-api-key` 驗證         | Token 外洩會失效，建議環境變數管理 |
| **OCR 文字辨識**  | 讀取圖片文字        | `easyocr.Reader`                     | OCR 文字可能不完整或誤讀，影響後續分析 |
| **文字解析**      | 解析成分列表、營養標示 | 自訂字串拆分 + regex                 | 文字格式差異大，需容錯處理 |
| **熱量計算**      | 根據蛋白質/脂肪/碳水換算總熱量 | 自訂函數                             | OCR 誤讀可能影響數值計算 |
| **過敏原判定**    | 與預設過敏原清單比對 | 列表比對                             | OCR 漏字會導致漏判 |
| **素食類別判定**  | 判斷全素、蛋奶素、非素食 | 字串比對成分                         | OCR 或資料不完整可能判定錯誤 |
| **標註圖片**      | 在圖片上標出文字區塊 | Pillow 畫矩形 & 文字                 | 字體/解析度問題可能影響標註清楚度 |
| **Metadata 讀取** | 取得檔案大小、EXIF 資訊、上傳時間 | Pillow + os.path + datetime          | 截圖或網路圖片可能無 EXIF |
| **Metadata 存檔** | 存入 Excel          | pandas / openpyxl                    | 多人同時上傳可能導致 Excel 檔案競爭 |
| **前端顯示**      | 顯示 JSON 資訊 + 標註圖片 | HTML + Bootstrap + JS                | 大量圖片可能影響前端渲染性能 |
| **外網測試**      | ngrok 暴露本地服務  | ngrok CLI                            | 連線速率受限於本地網路與 ngrok 流量 |

## 4. 技術實作程度

| 模組                    | 技術成熟度 / 開發注意事項                          |
|-------------------------|-----------------------------------------------------|
| **FastAPI**             | 已穩定，用於小型 REST API 足夠；可輕鬆加入 Token 驗證 |
| **easyocr**             | 對中文英文都有支援，對清晰的食品標示效果好；文字解析需容錯 |
| **Pillow**              | 標註圖片與讀 EXIF 都可；需要注意文字位置與字體大小 |
| **pandas / openpyxl**   | Excel 存檔簡單、快速；多人同時寫入需注意檔案鎖或改 CSV/SQLite |
| **Bootstrap + JS**      | 前端簡單上傳介面即可；不需額外框架 |
| **ngrok**               | 測試方便，但非長期部署方案；外網使用需注意安全性 |
| **Token 驗證**          | HTTP Header x-api-key 實作簡單，可防止非授權使用 |

## 5. 專案設計原則

- **分層清楚**
  - API 層 → Controller/Endpoint 角色
  - Service 層 → 業務邏輯處理
  - Model 層 → 資料結構
  - Utils → 檔案/Metadata 輔助函數

- **可維護性與可測試性**
  - Service 層可獨立單元測試
  - Utils 層可重複使用

- **簡單前端**
  - HTML + Bootstrap 足夠
  - JS 負責圖片上傳、結果顯示

- **安全與容錯**
  - 檔案檢查、Token 驗證
  - OCR/文字解析容錯
  - Metadata 讀取失敗不影響主要服務

## 6. 後續擴充方向

- 支援多張圖片批次上傳（如果有時間）
- 儲存結果到資料庫（SQLite/MySQL）替代 Excel（更專業一點）
- 增加 LINE Bot 前端（有趣的整合）
- 支援語音/文字輸出多媒體回傳（進階功能）
- 前端進一步優化，提供圖文摘要或 CSV 下載（如果用戶反饋好）

這份文件可作為 Side Project 技術紀錄 + 相似專案討論模板，後續新增功能或改架構都可以依此對照。開發過程將學到很多 OCR 和 API 設計的知識，挺有趣的！🚀