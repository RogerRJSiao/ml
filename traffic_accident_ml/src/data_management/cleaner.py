"""
交通事故資料集清洗腳本
    接續 merger.py 的結果，將 data/processed/merged_years/ 年度彙總檔清洗之後，
    輸出到 data/processed/cleaned/，預計交付給 loader.py 彙總成訓練前的資料集。

使用方式：
    python -m src.data_management.cleaner
    python -m src.data_management.cleaner --force 2021 2022   #--強制重新清洗指定西元年

規則：
- 一個年度一個年度分開清洗，不跨年度合併：
  data/processed/merged_years/TW_traffic_accident_Y<年度>.csv
  -> data/processed/cleaned/TW_traffic_accident_Y<年度>_cleaned.csv
- 被移除的資料列統一存成 data/processed/cleaned/TW_traffic_accident_Y<年度>_removed.csv：
  第一欄 reason 為移除原因（日期時間異常、整列完全重複、A1/A2重複（保留A1）、當事者順位重複、年齡登錄異常、
  經緯度異常、缺少當事者順位、當事者順位3以後（棄用）），其餘欄位為原始資料列原貌。
- 欄位對照表、編碼對照表為跨年度總表（含「年度」欄），每次執行由全部年度重新產生：
  data/processed/cleaned/TW_traffic_accident_欄位對照表.csv、TW_traffic_accident_編碼對照表.csv
- 用 registry/cleaned_years.json 記錄每個年度的來源檔 sha256，
  來源檔未變更且輸出檔都在時略過該年度，總表中該年度的列沿用上次結果。
  以 --force 指定西元年時，只處理該年度且不論狀態一律重新清洗，總表中其他年度的列沿用上次結果。
- 欄位名稱、各欄位的處理方式與清洗規則常數皆定義於 cleaning_rules.py：
  每個欄位的層級（案件/當事者）、處理方式（KEEP/DROP/SPLIT/MERGE/INTERNAL/DERIVE）
  與輸出英文欄名以 COLUMN_MIGRATION 為準，下列步驟中的大寫名稱皆為其中的常數。

處理步驟：
    1. 以字串讀入全部欄位，並檢查原始欄位是否齊全（RAW_COLUMNS）。刪除檔尾說明列。
    2. 檢查資料型別
       (a) 發生日期、發生時間亦維持字串，並檢查格式與合理性；異常者同一案件的所有資料列一併刪除。
    3. 去除重複並整併案件：（案件鍵值 CASE_KEY：發生日期+發生時間+處理單位名稱警局層+經度+緯度+發生地點）
       (a) 整列完全相同者只留一筆
       (b) 同一案件同時出現在 A1 與 A2 時，保留 A1、刪除 A2。
    4. 刪除不合理案件：（同一案件的所有當事者資料列一併刪除）
       (a) 同一案件中，當事者順位有兩筆以上相同者，視為登錄有問題。
       (b) 任一當事者「當事者事故發生時年齡」超過 100。
       (c) 經度、緯度不在台澎金馬範圍（GEO_BOXES）或空白。
       (d) 缺少當事者順位 1 或 2（例如只有一筆當事者資料），無法展開成兩個當事者。
    5. 轉為「一起案件一列」：
       (a) 只保留當事者順位 1、2 的資料列（順位 3 以後棄用，記入移除資料檔）
       (b) 以案件鍵值驗證每起案件皆恰有順位 1、2
       (c) 當事者層級欄位依順位展開為 順位1_欄位名稱…、順位2_欄位名稱…（PARTY_PREFIXES），
           並移除當事者順位欄（INTERNAL）
    6. 數值欄位（NUMERIC_COLS）轉為數值型態；步驟 7 起空白一律以缺失值表示
    7. 產生衍生欄位（DERIVE）：
       (a) 由發生地點解析 縣市、縣市鄉鎮市區（TOWN_PATTERN），
           並判定 高快速公路附近 Y/N（HIGHWAY_PATTERN、HIGHWAY_POLICE）
       (b) 死亡受傷人數拆成 死亡人數、受傷人數（SPLIT）
       (c) 合併 當事者行動狀態大類別名稱 + 子類別名稱 → 當事者行動狀態（MERGE）
       (d) 依發生時間建立時段（PERIOD_BINS），並記錄時段的編碼對照（併入步驟 11 的編碼對照表）
    8. 資料內容校正：
       (a) 修正錯字（TYPO_FIX）
       (b) 依 當事者屬-性-別名稱 判定，空白的當事者類別欄位（PARTY_CAT_COLS）補填標籤：
           非人類當事者（NONHUMAN）年齡標 0、補填「not_applicable」（NA_LABEL）；
           肇逃未查獲（UNSOLVED）年齡維持原值、補填「unsolved」（UNSOLVED_LABEL）
       (c) 「當事者事故發生時年齡」為 -1 的當事者視同無或物
            年齡 0、空白的當事者類別欄位補填「not_applicable」（NA_LABEL）
    9. 去除不使用的欄位（DROP_COLS，即 COLUMN_MIGRATION 中標記 DROP 者），並調整欄位順序。
   10. 類別欄位改為英文代碼（CATEGORY_MAPS）
   11. 欄位改為英文名稱（CASE_RENAME、PARTY_RENAME，由 COLUMN_MIGRATION 推導），
       並產生該年度的欄位對照表（含每欄填答樣式，依頻度由高到低）
       及編碼對照表（每個類別欄位的原始值 → 英文代碼），全部年度跑完後合併成總表輸出
"""
import argparse
from datetime import datetime, timezone

import pandas as pd

from .common import BASE_DIR, load_json_record, save_json_record, sha256_of_file
from .cleaning_rules import (
    #--原始欄位（依原始欄位順序）
    COL_YEAR, COL_MONTH, COL_INCIDENT_DATE, COL_INCIDENT_TIME, COL_SEVERITY, COL_POLICE,
    COL_ADDRESS, COL_CASUALTY, COL_PARTY_ORDER, COL_GENDER, COL_AGE, COL_PROTECTION,
    COL_STATUS_MAJOR, COL_STATUS_MINOR, COL_LONGITUDE, COL_LATITUDE,
    #--衍生欄位
    COL_PERIOD, COL_IS_NEAR_HIGHWAY, COL_CITY, COL_CITY_DISTRICT,
    COL_DEAD_NUM, COL_INJURED_NUM, COL_STATUS,
    #--欄位 migration 推導結果、案件鍵值
    RAW_COLUMNS, DROP_COLS, CASE_RENAME, PARTY_RENAME, CASE_KEY, PARTY_PREFIXES,
    #--資料內容校正、轉型與類別代碼
    NONHUMAN, NA_LABEL, UNSOLVED, UNSOLVED_LABEL, PARTY_CAT_COLS,
    TYPO_FIX, NUMERIC_COLS, CATEGORY_MAPS, PERIOD_BINS,
    GEO_BOXES,
    #--地點解析
    TOWN_PATTERN, HIGHWAY_PATTERN, HIGHWAY_POLICE,
)

#--資料集的存放位置
#--年度彙總資料集目錄
MERGED_YEARS_DIR = BASE_DIR / "data" / "processed" / "merged_years"
#--清洗後資料集目錄、檔案狀態中繼資料
CLEANED_DIR = BASE_DIR / "data" / "processed" / "cleaned"
CLEANED_RECORD_PATH = BASE_DIR / "registry" / "cleaned_years.json"
#--跨年度總表
DICT_PATH = CLEANED_DIR / "TW_traffic_accident_欄位對照表.csv"
CODEBOOK_PATH = CLEANED_DIR / "TW_traffic_accident_編碼對照表.csv"

#--原始資料列id的欄名：回查原始資料被移除的列，輸出前會移除該欄
RID = "_rid"

def log(msg):
    print(msg)


def read_one(path):
    """步驟 1：以字串讀入、檢查原始欄位是否齊全（RAW_COLUMNS）、刪除檔尾說明列"""
    #--讀取csv
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    #--整理欄名
    df.columns = df.columns.str.strip()
    #--檢查欄名是否完整
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少原始欄位（請對照 COLUMN_MIGRATION）：{missing}")
    
    #--計算有效讀入列數，不包括第一欄不是年度的資料
    raw_row_count = len(df)
    is_data_row = df[COL_YEAR].str.fullmatch(r"\d{4}")
    df = df[is_data_row].reset_index(drop=True)
    log(f"[讀入] {path}: {raw_row_count} 列，刪除說明列 {raw_row_count - len(df)} 列")
    #--新增一欄，建立原始資料的索引值
    df[RID] = df.index
    return df, raw_row_count

def check_datetime(df, removed):
    """步驟 2：檢查日期、時間字串格式，異常者整起案件刪除"""
    #--去除前後空白；時間左側補 0 至 6 碼
    df[COL_INCIDENT_DATE] = df[COL_INCIDENT_DATE].str.strip()
    df[COL_INCIDENT_TIME] = df[COL_INCIDENT_TIME].str.strip()
    #--補 0 前先記下空白，避免被補成 000000 而視為正常
    is_blank_time = df[COL_INCIDENT_TIME] == ""         
    df[COL_INCIDENT_TIME] = df[COL_INCIDENT_TIME].str.zfill(6) #--補0

    #--日期：無法依 YYYYMMDD 解析者視為異常
    is_bad_date = pd.to_datetime(df[COL_INCIDENT_DATE], format="%Y%m%d", errors="coerce").isna()
    #--時間：拆成時、分、秒，空白、非 6 位數字或超出範圍者視為異常
    t = df[COL_INCIDENT_TIME]
    hh, mm, ss = (pd.to_numeric(t.str[i:i + 2], errors="coerce") for i in (0, 2, 4))
    is_bad_time = is_blank_time | ~t.str.fullmatch(r"\d{6}") | (hh > 23) | (mm > 59) | (ss > 59)
    log(f"[檢查] 日期格式異常 {is_bad_date.sum()} 列；時間格式異常 {is_bad_time.sum()} 列")
    #--同一案件任一列異常即整起刪除（移除原因記入 *_removed.csv）用索引JOIN
    is_bad = (is_bad_date | is_bad_time).groupby([df[c] for c in CASE_KEY], sort=False).transform("any")
    return drop_cases(df, is_bad, removed, "日期時間異常", "日期或時間格式異常的案件", unit="起")

def parse_location(case):
    """case：去除重複後的案件表，回傳 縣市、鄉鎮市區、高快速公路附近"""
    loc = case[COL_ADDRESS]
    out = pd.DataFrame(index=case.index)
    out[COL_CITY] = loc.str[:3]
    out[COL_CITY_DISTRICT] = out[COL_CITY] + loc.str[3:].str.extract(TOWN_PATTERN)[0]
    is_near_highway = loc.str.contains(HIGHWAY_PATTERN, regex=True, na=False) | (case[COL_POLICE] == HIGHWAY_POLICE)
    out[COL_IS_NEAR_HIGHWAY] = is_near_highway.map({True: "Y", False: "N"})
    return out


MAX_LIST = 30   # 不同值超過此數量時，只列出前 MAX_LIST 名


def fmt_value(v):
    if pd.isna(v) or v == "":
        return "(空白)"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def build_dictionary(df, rename):
    """欄位對照表：英文欄位、中文欄位、資料型態、不同值數量、空白筆數、數值範圍、填答樣式（依頻度高到低）"""
    rows, n = [], len(df)
    for c in df.columns:
        s = df[c]
        vc = s.value_counts(dropna=False)
        items = [f"{fmt_value(v)}（{cnt}，{cnt / n:.1%}）" for v, cnt in vc.head(MAX_LIST).items()]
        if len(vc) > MAX_LIST:
            items.append(f"…其餘 {len(vc) - MAX_LIST} 種")
        is_num = pd.api.types.is_numeric_dtype(s)
        rows.append({
            "英文欄位": rename[c],
            "中文欄位": c,
            "資料型態": "數值" if is_num else "文字",
            "不同值數量": int(s.nunique(dropna=True)),
            "空白筆數": int(s.isna().sum()),
            "數值範圍": f"{fmt_value(s.min())} ~ {fmt_value(s.max())}" if is_num else "",
            "填答樣式（依出現次數由高到低，括號內為筆數、比例）": "；".join(items),
        })
    return pd.DataFrame(rows)


def build_codebook(code_rows, rename):
    """編碼對照表：每個類別欄位的「原始值 → 英文代碼」，含筆數（依英文代碼、筆數排序）"""
    cb = pd.DataFrame(code_rows)
    def en_name(zh):
        if zh in rename:
            return rename[zh]
        if f"順位1_{zh}" in rename:
            return rename[f"順位1_{zh}"].replace("party1_", "party1/2_")
        return ""
    cb.insert(0, "英文欄位", cb["中文欄位"].map(en_name))
    order = {c: i for i, c in enumerate(dict.fromkeys(cb["中文欄位"]))}
    cb["_o"] = cb["中文欄位"].map(order)
    cb["_g"] = cb.groupby(["中文欄位", "英文代碼"])["筆數"].transform("sum")
    cb = cb.sort_values(["_o", "_g", "英文代碼", "筆數"], ascending=[True, False, True, False])
    cb = cb.drop(columns=["_o", "_g"]).rename(columns={"筆數": "筆數（案件欄位為案件數；當事者欄位為順位1+2人數合計）"})
    return cb


def get_removed_path(out_path):
    """移除資料檔路徑：TW_traffic_accident_Y<年度>_removed.csv"""
    return out_path.with_name(out_path.name.replace("_cleaned.csv", "_removed.csv"))


def drop_rows(df, is_bad, removed, reason):
    """
    刪除 is_bad 標記的列，並把（移除原因, 原始列編號）加入 removed
    輸入：
        df：資料集（需含 RID 欄）
        is_bad：要刪除的列，布林遮罩（Series 或 ndarray），長度與 df 相同
        removed：移除紀錄 list，直接 append (reason, 被刪列的 RID)
        reason：移除原因，寫入 *_removed.csv 的 reason 欄
    回傳刪除後的資料集（索引重設為 0..n-1）
    """
    removed.append((reason, df.loc[is_bad, RID]))
    return df[~is_bad].reset_index(drop=True)

def drop_cases(df, is_bad, removed, reason, label, unit="組", extra=""):
    """刪除 is_bad 標記的列（整組案件），並記錄移除原因"""
    n_case = df.loc[is_bad, CASE_KEY].drop_duplicates().shape[0]
    log(f"[刪除] {label} {n_case} {unit}、共 {is_bad.sum()} 列，整組刪除{extra}")
    return drop_rows(df, is_bad, removed, reason)


def dedupe_cases(df, removed):
    """步驟 3：去除重複並整併案件"""
    #--(a) 排除_rid之後，才檢查是否有重複列資料
    is_dup = df.drop(columns=RID).duplicated()
    df = drop_rows(df, is_dup, removed, "整列完全重複")
    log(f"[去重] 整列完全重複 {is_dup.sum()} 列已刪除，剩 {len(df)} 列")

    #--(b) 多次通報的同一案件同時出現在 A1 與 A2 時，保留 A1、刪除 A2
    #--A1/A2 比對鍵值：比 CASE_KEY 寬鬆，不含警局與經緯度
    a1a2_key = [COL_INCIDENT_DATE, COL_INCIDENT_TIME, COL_ADDRESS]
    is_a1 = df[COL_SEVERITY] == "A1"
    is_a2 = df[COL_SEVERITY] == "A2"
    a1_keys = df.loc[is_a1, a1a2_key].drop_duplicates()
    is_in_a1 = df[a1a2_key].merge(a1_keys.assign(_a1=1), on=a1a2_key, how="left")["_a1"].notna().to_numpy()
    is_a2_overlap = is_a2.to_numpy() & is_in_a1
    log(f"[A1/A2] 刪除同時出現在 A1 的 A2 資料 {is_a2_overlap.sum()} 列")
    return drop_rows(df, is_a2_overlap, removed, "A1/A2重複（保留A1）")

def split_casualties(df):
    """步驟 7(b)：死亡受傷人數拆成 死亡人數、受傷人數（放在原欄位位置）"""
    x = df[COL_CASUALTY].str.extract(r"死亡(\d+);受傷(\d+)")
    if x.isna().any().any():
        log(f"[警告] 死亡受傷人數無法解析 {x.isna().any(axis=1).sum()} 列")
    pos = df.columns.get_loc(COL_CASUALTY)
    df.insert(pos, COL_DEAD_NUM, pd.to_numeric(x[0]))
    df.insert(pos + 1, COL_INJURED_NUM, pd.to_numeric(x[1]))
    return df.drop(columns=COL_CASUALTY)


def drop_dup_orders(df, removed):
    """步驟 4(a)：同一案件內當事者順位重複者，整組刪除"""
    is_dup = df.groupby(CASE_KEY, sort=False)[COL_PARTY_ORDER].transform(
        lambda s: s.duplicated().any()).astype(bool)
    return drop_cases(df, is_dup, removed, "當事者順位重複", "同一案件內當事者順位重複者")

def drop_bad_ages(df, removed):
    """步驟 4(b)：任一當事者年齡 >100 的案件，整組刪除"""
    age = pd.to_numeric(df[COL_AGE], errors="coerce")
    is_bad_row = age > 100
    is_bad_age = is_bad_row.groupby([df[c] for c in CASE_KEY], sort=False).transform("any")
    return drop_cases(df, is_bad_age, removed, "年齡登錄異常", "年齡 >100 的案件",
                      extra=f"（觸發列 {is_bad_row.sum()} 列）")

def drop_out_of_bounds(df, removed):
    """步驟 4(c)：經緯度不在台澎金馬範圍（或空白）的案件，整起刪除"""
    lon = pd.to_numeric(df[COL_LONGITUDE], errors="coerce")
    lat = pd.to_numeric(df[COL_LATITUDE], errors="coerce")
    is_inside = pd.Series(False, index=df.index)
    for x0, x1, y0, y1 in GEO_BOXES.values():
        is_inside |= lon.between(x0, x1) & lat.between(y0, y1)
    return drop_cases(df, ~is_inside, removed, "經緯度異常",
                      "經緯度不在台澎金馬範圍（或空白）的案件", unit="起")

def drop_missing_parties(df, removed):
    """步驟 4(d)：缺少當事者順位 1 或 2 的案件（例如只有一筆當事者資料），整起刪除"""
    order = df[COL_PARTY_ORDER].str.strip()
    grp = [df[c] for c in CASE_KEY]
    has_1 = (order == "1").groupby(grp, sort=False).transform("any")
    has_2 = (order == "2").groupby(grp, sort=False).transform("any")
    return drop_cases(df, ~(has_1 & has_2), removed, "缺少當事者順位",
                      "缺少當事者順位 1 或 2 的案件", unit="起")

def drop_bad_cases(df, removed):
    """步驟 4：刪除不合理案件（順位重複、年齡異常、經緯度範圍外、缺少順位 1 或 2）"""
    df = drop_dup_orders(df, removed)
    df = drop_bad_ages(df, removed)
    df = drop_out_of_bounds(df, removed)
    return drop_missing_parties(df, removed)

def write_removed_to_csv(raw, removed, path):
    """
    被移除的原始資料列，統一存成一份

    removed = [("整列完全重複", [7]), ("當事者順位重複", [20, 21]), ("年齡登錄異常", [55, 56, 57])]
    
    out = 
    _rid  發生年度  發生日期  ...  reason
    7     2021     20210105  ...  整列完全重複
    20    2021     20210110  ...  當事者順位重複
    21    2021     20210110  ...  當事者順位重複
    55    2021     20210201  ...  年齡登錄異常
    56    2021     20210201  ...  年齡登錄異常
    57    2021     20210201  ...  年齡登錄異常
    """
    #--取得最原始資料集與_rid，並把_rid設為索引
    raw = raw.set_index(RID)
    #--查詢_rid，並增加reason欄位，再用合併pd.concat()把[]整理成.to_csv()可用格式
    out = pd.concat([raw.loc[rids].assign(reason=reason) for reason, rids in removed])
    #--寫入removed.csv
    out[["reason", *raw.columns]].to_csv(path, index=False, encoding="utf-8-sig")
    counts = "、".join(f"{reason} {len(rids)} 列" for reason, rids in removed)
    log(f"[移除資料] {path}: 共 {len(out)} 列（{counts}）")

def party_cols(col):
    """當事者欄位展開後的名稱：順位1_<col>、順位2_<col>"""
    return [p + col for p in PARTY_PREFIXES]


def resolve_cols(df, cols):
    """把欄位清單對應到 df 的實際欄位：案件欄位維持原名，當事者欄位換成 順位1_/順位2_ 兩欄"""
    out = []
    for c in cols:
        out += [c] if c in df.columns else [pc for pc in party_cols(c) if pc in df.columns]
    return out


def filter_and_combine_order_1_and_2(df, removed):
    """步驟 5：只保留當事者順位 1、2，驗證後展開成一起案件一列，並移除當事者順位欄與原始列編號"""
    #--(a) 只保留當事者順位 1、2 的資料列；順位 3 以後棄用，記入 removed
    is_kept = df[COL_PARTY_ORDER].str.strip().isin(["1", "2"])
    log(f"[篩選] 移除當事者順位 3 以後的資料 {(~is_kept).sum()} 列")
    df = drop_rows(df, ~is_kept, removed, "當事者順位3以後（棄用）")
    #--此後不再刪列，移除原始列編號（避免被當成當事者欄位展開）
    df = df.drop(columns=RID)

    #--(b) 驗證每起案件恰有順位 1、2，且順位前的案件層級欄位在同一案件內一致（不寫入輸出）
    grp = df.groupby(CASE_KEY, dropna=False, sort=False)
    is_valid_case = grp[COL_PARTY_ORDER].apply(lambda s: sorted(s.str.strip().astype(int)) == [1, 2])
    log(f"[驗證] 案件數 {len(is_valid_case)}，恰有順位 1、2 者 {is_valid_case.sum()}，異常 {(~is_valid_case).sum()}")
    check_cols = list(df.columns[: df.columns.get_loc(COL_PARTY_ORDER)])
    other = [c for c in check_cols if c not in CASE_KEY]
    is_inconsistent = (grp[other].nunique(dropna=False) > 1)
    if is_inconsistent.any().any():
        n_incons = is_inconsistent.sum()
        has_incons = n_incons > 0
        log(f"[驗證] 同一案件內其他案件層級欄位不一致：{n_incons[has_incons].to_dict()}")
    else:
        log("[驗證] 同一案件內，當事者順位前的所有欄位皆一致")

    #--(c) 依當事者順位展開成一起案件一列，並移除當事者順位欄
    #--順位前的欄位為案件層級，順位後為當事者層級（原始檔的經度、緯度排在順位後，但屬案件鍵值）
    k = df.columns.get_loc(COL_PARTY_ORDER)
    after = list(df.columns[k + 1:])
    case_cols = list(df.columns[:k]) + [c for c in after if c in CASE_KEY]
    cols = [c for c in after if c not in CASE_KEY]
    order = df[COL_PARTY_ORDER].str.strip()
    is_p1 = order == "1"
    is_p2 = order == "2"
    p1 = df[is_p1][case_cols + cols].rename(columns={c: f"順位1_{c}" for c in cols})
    p2 = df[is_p2][CASE_KEY + cols].rename(columns={c: f"順位2_{c}" for c in cols})
    df = p1.merge(p2, on=CASE_KEY, how="left", validate="one_to_one")
    p2_cols = [f"順位2_{c}" for c in cols]
    log(f"[展開] 依當事者順位展開：{len(df)} 起案件（一起一列），順位2 缺漏 "
        f"{df[p2_cols].isna().all(axis=1).sum()} 起")
    return df


def convert_numeric(df):
    """步驟 6：數值欄位轉為數值型態（空字串 → 缺失）"""
    df = df.replace("", pd.NA)
    for c in resolve_cols(df, NUMERIC_COLS):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def add_location(df):
    """步驟 7(a)：由發生地點解析縣市、鄉鎮市區、高快速公路附近（已是一起案件一列）"""
    loc = parse_location(df)
    df = pd.concat([df, loc], axis=1)
    log(f"[地點] {len(loc)} 起案件；鄉鎮市區無法解析 {loc[COL_CITY_DISTRICT].isna().sum()} 筆；"
        f"縣市 {loc[COL_CITY].nunique()} 種、縣市鄉鎮市區 {loc[COL_CITY_DISTRICT].nunique()} 種；"
        f"高快速公路附近=Y {(loc[COL_IS_NEAR_HIGHWAY] == 'Y').sum()} 筆")
    return df

def merge_action(df):
    """步驟 7(c)：合併行動狀態大/子類別為「當事者行動狀態」（順位1、2 各一欄）"""
    for p in PARTY_PREFIXES:
        major, minor = df[p + COL_STATUS_MAJOR].str.strip(), df[p + COL_STATUS_MINOR].str.strip()
        action = (major + "-" + minor).where(minor.notna(), major)
        df.insert(df.columns.get_loc(p + COL_STATUS_MAJOR), p + COL_STATUS, action)
        df = df.drop(columns=[p + COL_STATUS_MAJOR, p + COL_STATUS_MINOR])
    log(f"[欄位] 合併行動狀態大/子類別為「{COL_STATUS}」")
    return df

def add_period(df):
    """步驟 7(d)：依發生時間建立時段，並記錄時段的編碼對照，回傳 (df, 時段編碼對照紀錄)"""
    hour = df[COL_INCIDENT_TIME].str[:2].astype(int)
    period = pd.Series(pd.NA, index=df.index, dtype="object")
    period_rows = []
    for h0, h1, name in PERIOD_BINS:
        is_in_bin = (hour >= h0) & (hour < h1)
        period[is_in_bin] = name
        #--時段直接以英文代碼建立，不在 CATEGORY_MAPS 中，於此記錄對照；
        #--同名時段有多個區間（late_night、rush_hour），逐區間各記一列
        period_rows.append({"中文欄位": COL_PERIOD, "原始值": f"發生時間 {h0:02d}:00～{h1 - 1:02d}:59",
                            "英文代碼": name, "筆數": int(is_in_bin.sum())})
    df.insert(df.columns.get_loc(COL_INCIDENT_TIME) + 1, COL_PERIOD, period)
    return df, period_rows

def derive_columns(df):
    """步驟 7：衍生欄位（地點、死亡/受傷人數、當事者行動狀態、時段），回傳 (df, 時段編碼對照紀錄)"""
    df = add_location(df)
    df = split_casualties(df)
    df = merge_action(df)
    return add_period(df)


def correct_values(df):
    """步驟 8：資料內容校正（修正錯字、標記非人類當事者與年齡 -1 當事者；順位1、2 分別處理），
    判定完成後去除 當事者屬-性-別名稱"""
    for wrong, right in TYPO_FIX.items():
        n_fix = 0
        for c in party_cols(COL_PROTECTION):
            n_fix += df[c].str.contains(wrong, na=False).sum()
            df[c] = df[c].str.replace(wrong, right, regex=False)
        log(f"[錯字] {COL_PROTECTION}：「{wrong}」→「{right}」{n_fix} 人")

    #--非人類當事者：年齡標 0；肇逃未查獲：年齡維持原值。兩者空白類別欄位各自補填標籤
    fill_party_blanks(df, lambda p: df[p + COL_GENDER].isin(NONHUMAN),
                      NA_LABEL, "非人類當事者", zero_age=True)
    fill_party_blanks(df, lambda p: df[p + COL_GENDER].isin(UNSOLVED),
                      UNSOLVED_LABEL, "肇逃未查獲當事者")
    #--其餘年齡為 -1 者視同無或物：年齡標 0、空白類別欄位補填 NA_LABEL
    fill_party_blanks(df, lambda p: (df[p + COL_AGE] == -1) & ~df[p + COL_GENDER].isin(NONHUMAN + UNSOLVED),
                      NA_LABEL, "年齡 -1 當事者（視同無或物）", zero_age=True)
    return df.drop(columns=party_cols(COL_GENDER))


def fill_party_blanks(df, select, label, name, zero_age=False):
    """select(p) 選出的順位 p 當事者，空白的當事者類別欄位（PARTY_CAT_COLS）填入 label（順位1、2 分別處理）"""
    n_party = age_changed = n_fill = 0
    for p in PARTY_PREFIXES:
        is_hit = select(p).fillna(False).astype(bool)
        n_party += is_hit.sum()
        if zero_age:
            age_changed += (is_hit & (df[p + COL_AGE] != 0)).sum()
            df.loc[is_hit, p + COL_AGE] = 0
        for c in PARTY_CAT_COLS:
            is_blank = is_hit & df[p + c].isna()
            n_fill += is_blank.sum()
            df.loc[is_blank, p + c] = label
    age_msg = f"年齡設為 0（其中原本非 0 者 {age_changed} 人），" if zero_age else ""
    log(f"[{name}] {n_party} 人：{age_msg}空白類別欄位填入「{label}」{n_fill} 格")


def reorder_columns(df):
    """步驟 9：去除不使用的欄位，並調整欄位順序"""
    missing = [c for c in DROP_COLS if not resolve_cols(df, [c])]
    if missing:
        raise ValueError(f"找不到欄位：{missing}")
    df = df.drop(columns=resolve_cols(df, DROP_COLS))
    log(f"[欄位] 去除 {len(DROP_COLS)} 欄")
    front = [COL_YEAR, COL_MONTH, COL_INCIDENT_DATE, COL_INCIDENT_TIME, COL_PERIOD,
             COL_LONGITUDE, COL_LATITUDE, COL_SEVERITY, COL_POLICE, COL_IS_NEAR_HIGHWAY,
             COL_ADDRESS, COL_CITY, COL_CITY_DISTRICT]
    return df[front + [c for c in df.columns if c not in front]]


def encode_categories(df):
    """步驟 10：類別欄位改為英文代碼，回傳 (df, 編碼對照紀錄)"""
    code_rows = []
    for col, mp in CATEGORY_MAPS.items():
        cols = resolve_cols(df, [col])               # 當事者欄位：順位1、2 合計
        values = pd.concat([df[c] for c in cols])
        vc = values.value_counts()
        for orig in list(mp) + [v for v in vc.index if v not in mp]:
            code_rows.append({"中文欄位": col, "原始值": orig,
                              "英文代碼": mp.get(orig, "(未定義，維持原值)"),
                              "筆數": int(vc.get(orig, 0))})
        is_unknown = values.notna() & ~values.isin(mp)
        unknown = values[is_unknown].unique()
        if len(unknown):
            log(f"[警告] {col} 有未定義對應的值，維持原值：{list(unknown)}")
        for c in cols:
            df[c] = df[c].map(lambda v: mp.get(v, v))
    log(f"[代碼] 已改為英文代碼：{'、'.join(CATEGORY_MAPS)}")
    return df, code_rows


def export(df, code_rows, out_path):
    """步驟 11：把原始欄位名稱改為英文，輸出清洗結果到csv，也回傳兩個變數：欄位對照表, 編碼對照表"""
    #--檢查輸出欄名是否登錄在CASE_RENAME、PARTY_RENAME
    rename = dict(CASE_RENAME)
    for i in (1, 2):
        rename.update({f"順位{i}_{zh}": f"party{i}_{en}" for zh, en in PARTY_RENAME.items()})
    unmapped = [c for c in df.columns if c not in rename]
    if unmapped:
        raise ValueError(f"以下欄位沒有英文名稱：{unmapped}")

    #--建立對照表
    dictionary = build_dictionary(df, rename)
    codebook = build_codebook(code_rows, rename)
    log(f"[欄位] 已改為英文名稱；編碼對照 {len(code_rows)} 列（併入總表）")    
    
    #--寫入cleaned.csv
    df = df.rename(columns=rename)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"[輸出] {out_path}: {len(df)} 列 × {df.shape[1]} 欄")

    return dictionary, codebook


def clean_year(src_path, out_path):
    """
    清洗單一年度彙總檔 src_path，輸出到 out_path
    輸入：年度彙總資料集路徑、年度清洗後資料集路徑
    回傳該年度的 (欄位對照表, 編碼對照表, 列數統計)
    列數統計含讀入列數、檔尾說明列數、輸出列數、各移除原因列數
    """
    removed = []
    #--步驟 1：以字串讀入並檢查原始欄位、刪除檔尾說明列
    raw, input_row_count = read_one(src_path)
    #--步驟 2：檢查日期、時間格式，異常案件整起刪除
    df = check_datetime(raw.copy(), removed)
    #--步驟 3：去除重複並整併案件
    df = dedupe_cases(df, removed)
    #--步驟 4：刪除不合理案件（順位重複、年齡異常、經緯度範圍外、缺少順位 1 或 2）
    df = drop_bad_cases(df, removed)
    #--步驟 5：轉為一起案件一列(只保留當事人順位1、2，順位 3 以後棄用)，並移除原始列編號
    df = filter_and_combine_order_1_and_2(df, removed)
    #--把步驟 2~5 移除/棄用的原始資料列，統一輸出
    write_removed_to_csv(raw, removed, get_removed_path(out_path))
    #--步驟 6：數值欄位轉為數值型態
    df = convert_numeric(df)
    #--步驟 7：產生衍生欄位（地點、死亡/受傷人數、當事者行動狀態、時段）
    df, period_rows = derive_columns(df)
    #--步驟 8：資料內容校正（修正錯字、補填當事者類別欄位）
    df = correct_values(df)
    #--步驟 9：去除不使用的欄位，並調整欄位順序
    df = reorder_columns(df)
    #--步驟 10：類別欄位改為英文代碼
    df, code_rows = encode_categories(df)
    #--步驟 11：欄位改為英文名稱並輸出，產生該年度的欄位對照表、編碼對照表
    dictionary, codebook = export(df, code_rows + period_rows, out_path)
    #--統計各移除原因的列數，寫入 cleaned_years.json 的 removed_counts
    removed_counts = {}
    for reason, rids in removed:
        removed_counts[reason] = removed_counts.get(reason, 0) + len(rids)
    counts = {
        "input_row_count": input_row_count,
        "footer_rows_dropped": input_row_count - len(raw),
        "output_row_count": len(df),
        "removed_counts": removed_counts,
    }
    return dictionary, codebook, counts


def write_summary_to_csv(tables, path, label):
    """各年度表加上「年度」欄後合併，整份覆寫 path"""
    total = pd.concat([t.assign(年度=year)[["年度", *t.columns]] for year, t in tables],
                      ignore_index=True)
    total.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"[總表] {label} {path}: {len(total)} 列（{len(tables)} 個年度）")


def read_summary_from_csv(path):
    """讀回上次輸出的總表"""
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def summarize_cols(total, year):
    """取出總表中該年度的欄名"""
    #--檢查是否為首次執行。首次執行無資料，回傳None
    if total is None:
        return None
    #--若非首次執行，可取出指定年度的欄名
    is_year = total["年度"] == year
    rows = total[is_year]
    return rows.drop(columns="年度").reset_index(drop=True) if len(rows) else None


def clean_all(force_years=()):
    """
    逐年清洗 data/processed/merged_years/ 底下的年度匯總檔
    - 回傳本次有重新清洗的輸出檔路徑清單。
    - 來源檔未變更、輸出檔與總表中該年度的列都在時，略過該年度。
    - 若指定 force_years 時只處理該些西元年，一律重新清洗；總表中其他年度的列沿用上次結果。
    """
    #--建立清洗後資料夾
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    #--檢查年度匯總檔csv
    force_years = sorted({str(y) for y in force_years})
    if force_years:
        #--指定西元年時，只取該年度的年度匯總檔，其他年度不列入
        sources = [MERGED_YEARS_DIR / f"TW_traffic_accident_Y{y}.csv" for y in force_years]
        missing = [p.name for p in sources if not p.exists()]
        if missing:
            log(f"[warn] --force 指定的年度找不到年度匯總檔：{', '.join(missing)}")
        sources = [p for p in sources if p.exists()]
        if not sources:
            return []
    else:
        sources = sorted(MERGED_YEARS_DIR.glob("TW_traffic_accident_Y*.csv"))
    if not sources:
        log(f"[skip] {MERGED_YEARS_DIR} 找不到年度匯總檔，請先執行 merger.py")
        return []
    
    #--取得資料集狀態中繼資料
    cleaned_record = load_json_record(CLEANED_RECORD_PATH)
    #--取得欄位對照表、編碼對照表csv
    old_dicts, old_codebooks = read_summary_from_csv(DICT_PATH), read_summary_from_csv(CODEBOOK_PATH)

    #--逐年檢查
    outputs, dicts, codebooks = [], [], []
    for src_path in sources:
        year = src_path.stem.rsplit("_Y", 1)[-1] #--取西元年
        out_path = CLEANED_DIR / f"{src_path.stem}_cleaned.csv"
        #--計算hash，並檢查檔案狀態是否可略過
        src_hash = sha256_of_file(src_path)
        previous = cleaned_record.get(f"Y{year}")
        dictionary, codebook = summarize_cols(old_dicts, year), summarize_cols(old_codebooks, year)
        if (
            not force_years                                 #--未指定強制重新清洗
            and previous is not None                        #--資料集狀態中繼資料有資料
            and previous.get("source_sha256") == src_hash   #--資料集狀態中繼資料與之前一致
            and out_path.exists()                           #--清洗後保留資料集存在
            and get_removed_path(out_path).exists()         #--清洗後刪除資料集存在
            and dictionary is not None                      #--欄位對照表中，有該年度的可沿用
            and codebook is not None                        #--編碼對照表中，有該年度的可沿用
        ):
            log(f"[skip] {src_path.name}：來源檔未變更，略過重新清洗")
        else:
            forced = "（--force 強制重新清洗）" if force_years else ""
            log(f"===== {src_path.name} -> {out_path.name} {forced}=====")
            #--開始執行單一年度的資料清洗
            dictionary, codebook, counts = clean_year(src_path, out_path)
            #--建立單筆年度已清洗的中繼資料
            cleaned_record[f"Y{year}"] = {
                "source_file": src_path.name,
                "source_sha256": src_hash,
                "input_row_count": counts["input_row_count"],
                "footer_rows_dropped": counts["footer_rows_dropped"],
                "output_file": out_path.name,
                "output_row_count": counts["output_row_count"],
                "output_sha256": sha256_of_file(out_path),
                "removed_file": get_removed_path(out_path).name,
                "removed_counts": counts["removed_counts"],
                "cleaned_at": datetime.now(timezone.utc).isoformat(),
            }
            outputs.append(out_path)    #--檢查用
        #--不管有沒有清洗，每年都會存入一筆
        dicts.append((year, dictionary))
        codebooks.append((year, codebook))

    #--指定西元年時，總表中其他年度的列沿用上次結果，避免整份覆寫時遺失
    if force_years:
        for old, tables in ((old_dicts, dicts), (old_codebooks, codebooks)):
            if old is None:
                continue
            done = {year for year, _ in tables}
            for year in sorted(set(old["年度"]) - done):
                tables.append((year, summarize_cols(old, year)))
            tables.sort(key=lambda t: t[0])

    #--只有在真的執行清洗時，才重新產生總表、寫回 registry
    if outputs:
        write_summary_to_csv(dicts, DICT_PATH, "欄位對照表")
        write_summary_to_csv(codebooks, CODEBOOK_PATH, "編碼對照表")
        save_json_record(CLEANED_RECORD_PATH, cleaned_record)
    return outputs


if __name__ == "__main__":
    #--命令列參數：不帶參數時逐年檢查
    parser = argparse.ArgumentParser(description="指定西元年清洗年度匯總檔")
    #--可一次指定多個年度，只處理這些年度的重新清洗
    parser.add_argument(
        #--在 "--force" 後方至少接一個參數，每個參數值都是int
        "--force", nargs="+", type=int, default=[], metavar="YEAR",
        help="強制重新清洗指定的西元年(如 --force 2021 2024)",
    )
    args = parser.parse_args()

    #--執行資料清洗流程
    clean_all(force_years=args.force)
 