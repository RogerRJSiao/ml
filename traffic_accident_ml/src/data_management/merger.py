"""
接續 extractor.py 的結果，將 data/raw/Y<年度>/ 底下解壓縮出來的多份 csv
（A1、A2 各期資料）合併成單一年度匯總檔，輸出到 data/processed/merged_years/。

使用方式：
    python -m src.data_management.merger

規則：
- 只合併實際資料檔，略過政府資料開放平台附帶的中繼資料檔。
- 同一年度內各檔案的欄位需完全一致，合併後只保留一列標頭。
- 輸出檔名為 TW_traffic_accident_Y<西元年>.csv，存放在 data/processed/merged_years/。
- 用 registry/merged_years.json 記錄每個年度來源檔案的 sha256，避免來源未變更時重複合併。
"""

import csv
from datetime import datetime, timezone

from .common import BASE_DIR, load_json_record, save_json_record, sha256_of_file

#--解壓縮後csv存放位置（extractor.py 的輸出）
RAW_DIR = BASE_DIR / "data" / "raw"
#--年度匯總檔存放位置，屬於衍生資料，與 data/raw 的原始資料分開
#--（data/processed/cleaned/ 則預留給未來欄位層級跨年度清洗完成的資料集）
MERGED_YEARS_DIR = BASE_DIR / "data" / "processed" / "merged_years"
MERGED_RECORD_PATH = BASE_DIR / "registry" / "merged_years.json"

#--資料開放平台有附帶中繼資料檔，合併時需排除
EXCLUDED_FILENAMES = {"file.csv", "manifest.csv", "schema-file.csv"}

#--目前慣例每年應有 A1 x1 + A2 x12（依月份拆分）共13份資料檔
EXPECTED_FILE_COUNT = 13


def _merge_files_by_year(year_dir, record):
    """合併單一年度資料夾內的所有事故資料 csv，回傳輸出檔路徑；略過或無資料則回傳 None。"""
    #--資料夾名稱固定格式 "Y<西元年>"，取出年度字串
    year = year_dir.name[1:]
    output_path = MERGED_YEARS_DIR / f"TW_traffic_accident_Y{year}.csv"

    #--排除中繼資料檔。查無任何檔案時，回傳None
    source_files = sorted(
        p for p in year_dir.glob("*.csv")
        if p.name not in EXCLUDED_FILENAMES
    )
    if not source_files:
        print(f"[skip] {year_dir.name}：找不到可合併的資料檔")
        return None

    #--目前資料開放平台慣例為 A1 x1 + A2 依月份拆成12份，共13份；
    #--數量不同時，僅提醒可能有缺檔或多檔，需人工確認
    if len(source_files) != EXPECTED_FILE_COUNT:
        print(
            f"[warn] {year_dir.name}：找到 {len(source_files)} 個資料檔"
            f"（預期 {EXPECTED_FILE_COUNT} 個），請確認來源資料是否有缺漏或重複"
        )

    #--計算每個csv的sha256
    source_hashes = {p.name: sha256_of_file(p) for p in source_files}
    
    #--檢查合併後的csv檔案：用檔名+hash判斷來源是否有變更
    previous = record.get(year_dir.name)
    if (
        previous is not None
        and previous.get("source_files") == source_hashes
        and output_path.exists()
    ):
        print(f"[skip] {year_dir.name}：來源檔案未變更，略過重新合併")
        return None

    header = None
    row_count = 0
    with output_path.open("w", newline="", encoding="utf-8-sig") as out_f:
        writer = csv.writer(out_f)
        for src in source_files:
            #--原始檔為 utf-8 with BOM，統一用 utf-8-sig 讀取避免 BOM 混入欄位名稱
            with src.open("r", newline="", encoding="utf-8-sig") as in_f:
                reader = csv.reader(in_f)
                try:
                    file_header = next(reader)
                except StopIteration:
                    continue

                if header is None:
                    #--以第一份檔案的欄位當作合併後的唯一標頭
                    header = file_header
                    writer.writerow(header)
                elif file_header != header:
                    raise ValueError(
                        f"{src.name} 的欄位與同年度其他檔案不一致，請確認資料來源"
                    )

                for row in reader:
                    writer.writerow(row)
                    row_count += 1

    #--新增/更新這個年度的合併紀錄
    record[year_dir.name] = {
        "source_files": source_hashes,
        "expected_file_count": EXPECTED_FILE_COUNT,
        "actual_file_count": len(source_files),
        "output_file": output_path.name,
        "output_row_count": row_count,
        "output_sha256": sha256_of_file(output_path),
        "merged_at": datetime.now(timezone.utc).isoformat(),
    }

    print(f"[ok] {year_dir.name}：合併 {len(source_files)} 個檔案，共 {row_count} 筆 -> {output_path.name}")
    return output_path


def merge_all():
    """合併 data/raw 底下每個年度資料夾，回傳所有輸出檔路徑清單。"""
    MERGED_YEARS_DIR.mkdir(parents=True, exist_ok=True)
    record = load_json_record(MERGED_RECORD_PATH)
    outputs = []
    for year_dir in sorted(RAW_DIR.glob("Y*")):
        #--檢查解壓縮後的目錄是否存在
        if not year_dir.is_dir():
            continue
        #--把相同年份csv檔案合併成1份
        result = _merge_files_by_year(year_dir, record)
        if result is not None:
            outputs.append(result)

    #--只有在真的有合併到任何年度時，才寫回 registry
    if outputs:
        save_json_record(MERGED_RECORD_PATH, record)
    return outputs


if __name__ == "__main__":
    merge_all()
