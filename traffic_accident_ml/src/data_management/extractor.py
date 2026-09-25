"""
自動解壓縮 data/incoming/ 底下的 zip 檔案，依檔名中的年度解壓縮到 data/raw/Y<年度>/。

使用方式：
    python -m src.data_management.extractor

規則：
- 為方便跨年度資料合併，zip 檔名需包含年度：
    - 西元年：例如 accident_2020.zip、2020_traffic.zip
    - 民國年：例如 109年傷亡道路交通事故資料.zip，將取出 "109" 會轉換成西元年 "2020"
- 解壓縮後原始 zip 檔案保留在 data/incoming/，不會被刪除。
- 用 registry/extracted_zips.json 記錄每個 zip 的 sha256，避免重複解壓縮同一份未變更的 zip；
  若同名 zip 內容有變，才會重新解壓縮並覆蓋 data/raw/Y<年度>/ 內對應內容。
"""

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

#--不管當前檔案位置，指定子專案根目錄的絕對路徑
#--Path(__file__)是這個py位置，resolve()轉成絕對路徑，parents[0]到[2]往回推目錄位置
BASE_DIR = Path(__file__).resolve().parents[2]
#--原始下載zip存放位置
INCOMING_DIR = BASE_DIR / "data" / "incoming"
#--解壓縮後csv存放位置
RAW_DIR = BASE_DIR / "data" / "raw"
EXTRACTED_RECORD_PATH = BASE_DIR / "registry" / "extracted_zips.json"

#--預先建立RE樣式(減少多次解析)、民國年西元年切換基準
ROC_YEAR_PATTERN = re.compile(r"(\d{2,3})年")
WESTERN_YEAR_PATTERN = re.compile(r"(19|20)\d{2}")
ROC_TO_WESTERN_OFFSET = 1911


def _sha256_of_file(path):
    #--建立雜湊運算物件
    digest = hashlib.sha256()
    with path.open("rb") as f:
        #--逐塊讀取，每次往下讀取(至多)1MB，直到檔案結束
        #--iter()兩種寫法，單一參數是iter(list)，兩參數是iter(callable, sentinel)
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            #--分批餵入1MB的檔案內容，累加計算
            digest.update(chunk)
    return digest.hexdigest()   #--轉成十六進位字串輸出


def _parse_year(filename):
    """從檔名判斷西元年份，優先辨識民國年，找不到再退回找西元年。"""
    #--re.search()回傳結果，可用group(0)抓符合文字"109年"，group(1)抓捕獲群組()內的"109"

    #--檢查民國年
    roc_match = ROC_YEAR_PATTERN.search(filename)
    if roc_match:
        roc_year = int(roc_match.group(1))
        return str(roc_year + ROC_TO_WESTERN_OFFSET)    #--轉換成西元年
    #--檢查西元年
    western_match = WESTERN_YEAR_PATTERN.search(filename)
    return western_match.group(0) if western_match else None


def _load_record():
    #--用pathlib.Path讀取檔案
    if EXTRACTED_RECORD_PATH.exists():
        #--json.loads()把json字串轉成Py物件
        return json.loads(EXTRACTED_RECORD_PATH.read_text(encoding="utf-8"))
    return {}


def _save_record(record):
    #--用pathlib.Path寫入檔案
    EXTRACTED_RECORD_PATH.write_text(
        #--json.dumps()把Py物件轉成json字串
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def extract_all():
    """解壓縮 data/incoming/ 下方待處理的 zip，回傳有實際被解壓縮的檔名清單。"""
    #--取得最新的zip中繼資料
    record = _load_record()
    processed = []

    for zip_path in sorted(INCOMING_DIR.glob("*.zip")):
        #--計算zip的hash
        current_hash = _sha256_of_file(zip_path)
        #--取出完整檔名(不包括目錄路徑)
        previous = record.get(zip_path.name)
        #--檢查zip檔名相同且hash相同，表示檔案未變更，無須更新
        if previous is not None and previous.get("sha256") == current_hash:
            print(f"[skip] {zip_path.name}：內容未變更，略過解壓縮")
            continue
        #--檢查zip檔名是否有年份
        year = _parse_year(zip_path.name)
        if year is None:
            print(f"[warn] {zip_path.name}：檔名找不到民國年或西元年，請重新命名後再執行，已略過")
            continue

        #--建立zip解壓縮產出csv的位置(自動建立父目錄，目錄已存在時不報錯)
        dest_dir = RAW_DIR / f"Y{year}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        #--開始zip解壓縮到dest_dir
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)

        #--新增一筆中繼資料紀錄
        record[zip_path.name] = {
            "sha256": current_hash,
            "year": year,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }
        processed.append(zip_path.name)
        print(f"[ok] {zip_path.name} -> data/raw/Y{year}/")

    #--只有在真的有解壓縮到任何 zip 時，才寫回 registry
    if processed:
        _save_record(record)
    return processed


if __name__ == "__main__":
    extract_all()
