# 食品標示 OCR 分析系統

## 1. 開發目的 🎯

本專案是建構一個簡易 **OCR 圖片文字辨識** 服務，擷取 **食物包裝、營養標示** 上的資訊。服務將以 FastAPI 提供端點，讓使用者能上傳圖檔到前端網頁 (或者 LINE Bot)，分析並回傳該圖像的食品相關資訊。

## 2. 功能簡述 📲 
- 熱量計算
- 過敏原判定（比對過敏原清單）
- 素食類別判定（全素、蛋奶素、非素食等）
- 標註圖片上的文字（可選是否下載）
- 圖片 Metadata 存檔（Excel，用於後續資料追蹤）

## 3. 專案檔案夾結構 📂
```
food_label_ocr/
├── main.py                     #--FastAPI 入口
├── requirements.txt            #--Python 套件列表
├── uploads/                    #--圖檔儲存
│   ├── annotated/                # 標註下載
│   └── organized/                # 上傳整理
├── app/
│   ├── api/                    #--API 層
│   │   └── routes.py             # 處理 HTTP 請求、Token 驗證
│   ├── services/               #--services 業務邏輯層
│   │   ├── file_processing.py    # 圖檔管理
│   │   ├── ocr_processing.py     # OCR、文字解析、標註圖片
│   │   └── food_processing.py    # 熱量/過敏原/素食判定
│   ├── models/                 #--model 資料結構層(to be checked)
│   │   └── food_models.py        # Ingredient、Nutrition、FoodItem
│   └── utils/                  #--工具層
│       ├── file_utils.py         # 儲存配置、副檔名驗證、檔名命名
│       └── metadata_utils.py     # 讀取圖片 Metadata + Excel 存檔(to be checked)
└── static/
    └── index.html                # 前端網頁腳本
```
## 4. 技術實作程度 (to be checked)

| 模組                    | 技術成熟度 / 開發注意事項                          |
|-------------------------|-----------------------------------------------------|
| **FastAPI**             | 已穩定，用於小型 REST API 足夠；可輕鬆加入 Token 驗證 |
| **easyocr**             | 對中文英文都有支援，對清晰的食品標示效果好；文字解析需容錯 |
| **Pillow**              | 標註圖片與讀 EXIF 都可；需要注意文字位置與字體大小 |
| **pandas / openpyxl**   | Excel 存檔簡單、快速；多人同時寫入需注意檔案鎖或改 CSV/SQLite |
| **Bootstrap + JS**      | 前端簡單上傳介面即可；不需額外框架 |
| **ngrok**               | 測試方便，但非長期部署方案；外網使用需注意安全性 |
| **Token 驗證**          | HTTP Header x-api-key 實作簡單，可防止非授權使用 |

- 後續擴充方向
  - 支援多張圖片批次上傳（如果有時間）
  - 儲存結果到資料庫（SQLite/MySQL）替代 Excel（更專業一點）
  - 增加 LINE Bot 前端（有趣的整合）
  - 支援語音/文字輸出多媒體回傳（進階功能）
  - 前端進一步優化，提供圖文摘要或 CSV 下載（如果用戶反饋好）

## 5. 快速部署本地 🛠️
1. 開啟 Anaconda Prompt，建立並啟動新的 python 虛擬環境。
    ```
    (base) C:\Users\User>cd /d D:\your-project
    (base) D:\your-project>mkdir food_label_ocr
    (base) D:\your-project>cd food_label_ocr
    (base) D:\your-project\food_label_ocr>conda create --name food_label_ocr python=3.11.8
    (base) D:\your-project\food_label_ocr>conda activate food_label_ocr
    (food_label_ocr) D:\your-project\food_label_ocr>
    ```

2. 安裝本專案必要套件/模組。(請確認 VSCode 開啟 .py 時，右下角虛擬環境是在 `python 3.11.8 food_label_ocr`)
    ``` Anaconda Prompt
    (food_label_ocr) D:\your-project\food_label_ocr>pip install -U fastapi uvicorn[standard]
    (food_label_ocr) D:\your-project\food_label_ocr>pip install python-multipart
    ```
    或，用 `pip install -r` 單一指令安裝全部套件。 (to be checked)
    ```
    pip install -r requirements.txt
    ```

3. 檢查 API 的最小服務是否能順利啟用。
    ``` Anaconda Prompt
    #--檢查 uvicorn 服務 (以調用 main.py 的 app 實體為例)
    #--注意此腳本已無 import error 等報錯。
    (food_label_ocr) D:\your-project\food_label_ocr>uvicorn main:app --reload
    ```

## 6. 技術新里程 🚀
1. **實作 Python 物件導向**：根據調用情境，設計方法時選擇 instance method、class method、static method。以「圖檔OCR上傳+分析端點 API」建置為例，上傳圖檔方法是用靜態方法 (該物件本身無 cls 或 self)，裡面的目錄路徑是用類別方法 (確保路徑一致性)，不同的上載再以各自的實體方法產生 (避免參數交互汙染)。

2. **理解 Python 套件相依性**：執行 FastAPI、Uvicorn 完全不需依賴 CUDA 或 PyTorch，由於 Web API 層與 ML 推論層是解耦的，故不會有 CUDA 版本衝突問題，可直接在 Python 3.11 環境下，直接安裝最新版的 FastAPI、Uvicorn，快速以 `http://127.0.0.1:8000/docs#/` 實測 API 服務。(目前 FastAPI 官方最低需求是 3.9+)

## 7. 延伸閱讀 🔗
- FastAPI：https://fastapi.tiangolo.com/
  - [Migrate from Pydantic v1 to Pydantic v2](https://fastapi.tiangolo.com/how-to/migrate-from-pydantic-v1-to-pydantic-v2/)
- Python-Multipart：https://multipart.fastapiexpert.com/
- ngrok：https://ngrok.com/
- STEAM 教育學習網-使用 ngrok 服務：https://steam.oxxostudio.tw/category/python/example/ngrok.html