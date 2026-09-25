"""extractor.py 與 merger.py 共用的工具函式：子專案根目錄路徑、sha256 雜湊計算、JSON 紀錄讀寫。"""

import hashlib
import json
from pathlib import Path

#--不管當前檔案位置，指定子專案根目錄的絕對路徑
BASE_DIR = Path(__file__).resolve().parents[2]


def sha256_of_file(path):
    #--建立雜湊運算物件
    digest = hashlib.sha256()
    with path.open("rb") as f:
        #--逐塊讀取，每次往下讀取(至多)1MB，直到檔案結束
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_record(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_json_record(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
