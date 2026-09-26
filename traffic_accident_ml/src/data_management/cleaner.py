"""
交通事故資料集清洗腳本
    接續 merger.py 的結果，將 data/processed/merged_years/ 年度彙總檔清洗之後，
    輸出到 data/processed/cleaned/，預計交付給 loader.py 彙總成訓練前的資料集。

使用方式：
    python -m src.data_management.cleaner

規則：
- 一個年度一個年度分開清洗，不跨年度合併：
  data/processed/merged_years/TW_traffic_accident_Y<年度>.csv
  -> data/processed/cleaned/TW_traffic_accident_Y<年度>_cleaned.csv
- 被移除的資料列統一存成 data/processed/cleaned/TW_traffic_accident_Y<年度>_removed.csv：
  第一欄 reason 為移除原因（整列完全重複、A1/A2重複（保留A1）、當事者順位重複、年齡登錄異常、
  經緯度異常、缺少當事者順位），其餘欄位為原始資料列原貌。
- 欄位對照表、編碼對照表為跨年度總表（含「年度」欄），每次執行由全部年度重新產生：
  data/processed/cleaned/TW_traffic_accident_欄位對照表.csv、TW_traffic_accident_編碼對照表.csv
- 用 registry/cleaned_years.json 記錄每個年度的來源檔 sha256 與清洗規則檔（RULE_FILES）sha256，
  兩者皆未變更且輸出檔都在時略過該年度，總表中該年度的列沿用上次結果。
- 欄位名稱、各欄位的處理方式與清洗規則常數皆定義於 cleaning_rules.py：
  每個欄位的層級（案件/當事者）、處理方式（KEEP/DROP/SPLIT/MERGE/INTERNAL/DERIVE）
  與輸出英文欄名以 COLUMN_MIGRATION 為準，下列步驟中的大寫名稱皆為其中的常數。

處理步驟：
    1. 以字串讀入全部欄位，並檢查原始欄位是否齊全（RAW_COLUMNS）；
       發生日期、發生時間亦維持字串，並檢查格式與合理性。
    2. 刪除檔尾說明列：資料提供日期、事故類別。
    3. 去除重複並整併案件：（案件鍵值 CASE_KEY：發生日期+發生時間+處理單位名稱警局層+經度+緯度+發生地點）
       (a) 整列完全相同者只留一筆
       (b) 同一案件同時出現在 A1 與 A2 時，保留 A1、刪除 A2。
    4. 刪除不合理案件：（同一案件的所有當事者資料列一併刪除）
       (a) 同一案件中，當事者順位有兩筆以上相同者，視為登錄有問題。
       (b) 任一當事者「當事者事故發生時年齡」為 -1 或超過 100。
       (c) 經度、緯度不在台澎金馬範圍（GEO_BOXES）或空白。
       (d) 缺少當事者順位 1 或 2（例如只有一筆當事者資料），無法展開成兩個當事者。
    5. 轉為「一起案件一列」：
       (a) 只保留當事者順位 1、2 的資料列
       (b) 以案件鍵值驗證每起案件皆恰有順位 1、2（經步驟 4 後應無異常）
       (c) 當事者層級欄位依順位展開為 順位1_欄位名稱…、順位2_欄位名稱…（PARTY_PREFIXES），
           並移除當事者順位欄（INTERNAL）
    6. 數值欄位（NUMERIC_COLS）轉為數值型態；步驟 7 起空白一律以缺失值表示
    7. 產生衍生欄位（DERIVE）：
       (a) 由發生地點解析 縣市、縣市鄉鎮市區（TOWN_PATTERN），
           並判定 高快速公路附近 Y/N（HIGHWAY_PATTERN、HIGHWAY_POLICE）
       (b) 死亡受傷人數拆成 死亡人數、受傷人數（SPLIT）
       (c) 合併 當事者行動狀態大類別名稱 + 子類別名稱 → 當事者行動狀態（MERGE）
       (d) 依發生時間建立時段（PERIOD_BINS）
    8. 資料內容校正：
       (a) 修正錯字（TYPO_FIX）
       (b) 依 當事者屬-性-別名稱 判定，空白的當事者類別欄位（PARTY_CAT_COLS）補填標籤：
           非人類當事者（NONHUMAN）年齡標 0、補填「not_applicable」（NA_LABEL）；
           肇逃未查獲（UNSOLVED）年齡維持原值、補填「unsolved」（UNSOLVED_LABEL）
    9. 去除不使用的欄位（DROP_COLS，即 COLUMN_MIGRATION 中標記 DROP 者），並調整欄位順序。
   10. 類別欄位改為英文代碼（CATEGORY_MAPS）
   11. 欄位改為英文名稱（CASE_RENAME、PARTY_RENAME，由 COLUMN_MIGRATION 推導），
       並產生該年度的欄位對照表（含每欄填答樣式，依頻度由高到低）
       及編碼對照表（每個類別欄位的原始值 → 英文代碼），全部年度跑完後合併成總表輸出
"""
from datetime import datetime, timezone
from pathlib import Path

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

#--年度匯總檔（merger.py 的輸出）與清洗後資料集的存放位置
MERGED_YEARS_DIR = BASE_DIR / "data" / "processed" / "merged_years"
CLEANED_DIR = BASE_DIR / "data" / "processed" / "cleaned"
#--跨年度總表（每列含「年度」欄）
DICT_PATH = CLEANED_DIR / "TW_traffic_accident_欄位對照表.csv"
CODEBOOK_PATH = CLEANED_DIR / "TW_traffic_accident_編碼對照表.csv"
#--清洗紀錄；來源 sha256 優先沿用 merger.py 紀錄的 output_sha256，免得重算大檔
CLEANED_RECORD_PATH = BASE_DIR / "registry" / "cleaned_years.json"
MERGED_RECORD_PATH = BASE_DIR / "registry" / "merged_years.json"
#--內容變更時需重新清洗的程式檔
RULE_FILES = [Path(__file__), Path(__file__).with_name("cleaning_rules.py")]

RID = "_rid"   # 原始資料列編號：用來從原始資料取回被移除的列，輸出前移除


def log(msg):
    print(msg)


def read_one(path):
    """步驟 1~2：以字串讀入、檢查原始欄位是否齊全（RAW_COLUMNS）、刪除檔尾說明列"""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少原始欄位（請對照 COLUMN_MIGRATION）：{missing}")
    n0 = len(df)
    df = df[df[COL_YEAR].str.fullmatch(r"\d{4}")].reset_index(drop=True)
    df[RID] = df.index
    log(f"[讀入] {path}: {n0} 列，刪除說明列 {n0 - len(df)} 列")
    return df


def parse_location(case):
    """case：去除重複後的案件表，回傳 縣市、鄉鎮市區、高快速公路附近"""
    loc = case[COL_ADDRESS]
    out = pd.DataFrame(index=case.index)
    out[COL_CITY] = loc.str[:3]
    out[COL_CITY_DISTRICT] = out[COL_CITY] + loc.str[3:].str.extract(TOWN_PATTERN)[0]
    hw = loc.str.contains(HIGHWAY_PATTERN, regex=True, na=False) | (case[COL_POLICE] == HIGHWAY_POLICE)
    out[COL_IS_NEAR_HIGHWAY] = hw.map({True: "Y", False: "N"})
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


def side_path(out_path, name):
    """附帶輸出檔路徑：輸出檔名_<name>.csv"""
    return out_path.with_name(f"{out_path.stem}_{name}.csv")


def removed_path(out_path):
    """移除資料檔路徑：TW_traffic_accident_Y<年度>_removed.csv"""
    return out_path.with_name(out_path.name.replace("_cleaned.csv", "_removed.csv"))


def drop_rows(df, bad, removed, reason):
    """刪除 bad 標記的列，並把（移除原因, 原始列編號）加入 removed"""
    removed.append((reason, df.loc[bad, RID]))
    return df[~bad].reset_index(drop=True)


def drop_cases(df, bad, removed, reason, label, unit="組", extra=""):
    """刪除 bad 標記的列（整組案件），並記錄移除原因"""
    n_case = df.loc[bad, CASE_KEY].drop_duplicates().shape[0]
    log(f"[刪除] {label} {n_case} {unit}、共 {bad.sum()} 列，整組刪除{extra}")
    return drop_rows(df, bad, removed, reason)


def write_removed(raw, removed, path):
    """被移除的原始資料列統一存成一份，第一欄 reason 為移除原因"""
    raw = raw.set_index(RID)
    out = pd.concat([raw.loc[rids].assign(reason=reason) for reason, rids in removed])
    out[["reason", *raw.columns]].to_csv(path, index=False, encoding="utf-8-sig")
    counts = "、".join(f"{reason} {len(rids)} 列" for reason, rids in removed)
    log(f"[移除資料] {path}: 共 {len(out)} 列（{counts}）")


def dedupe_cases(raw, out_path, removed):
    """步驟 3：去除重複並整併案件（整列去重後先完成步驟 1 的日期時間檢查）"""
    df = raw.copy()
    dup = df.drop(columns=RID).duplicated()
    df = drop_rows(df, dup, removed, "整列完全重複")
    log(f"[去重] 整列完全重複 {dup.sum()} 列已刪除，剩 {len(df)} 列")
    df = check_datetime(df, out_path)
    return drop_a1_overlap(df, removed)


def check_datetime(df, out_path):
    """步驟 1（續）：日期、時間維持字串並檢查格式"""
    df[COL_INCIDENT_DATE] = df[COL_INCIDENT_DATE].str.strip()
    df[COL_INCIDENT_TIME] = df[COL_INCIDENT_TIME].str.strip().str.zfill(6)
    bad_date = pd.to_datetime(df[COL_INCIDENT_DATE], format="%Y%m%d", errors="coerce").isna()
    t = df[COL_INCIDENT_TIME]
    hh, mm, ss = (pd.to_numeric(t.str[i:i + 2], errors="coerce") for i in (0, 2, 4))
    bad_time = ~t.str.fullmatch(r"\d{6}") | (hh > 23) | (mm > 59) | (ss > 59)
    log(f"[檢查] 日期格式異常 {bad_date.sum()} 列；時間格式異常 {bad_time.sum()} 列")
    if bad_date.any() or bad_time.any():
        df.loc[bad_date | bad_time, [COL_INCIDENT_DATE, COL_INCIDENT_TIME]].to_csv(
            side_path(out_path, "日期時間異常"), encoding="utf-8-sig")
    return df


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
    dup = df.groupby(CASE_KEY, sort=False)[COL_PARTY_ORDER].transform(
        lambda s: s.duplicated().any()).astype(bool)
    return drop_cases(df, dup, removed, "當事者順位重複", "同一案件內當事者順位重複者")


def drop_bad_ages(df, removed):
    """步驟 4(b)：任一當事者年齡為 -1 或 >100 的案件，整組刪除"""
    age = pd.to_numeric(df[COL_AGE], errors="coerce")
    bad_row = (age == -1) | (age > 100)
    bad_age = bad_row.groupby([df[c] for c in CASE_KEY], sort=False).transform("any")
    df = drop_cases(df, bad_age, removed, "年齡登錄異常", "年齡為 -1 或 >100 的案件",
                    extra=f"（觸發列 {bad_row.sum()} 列，其中 -1：{(age == -1).sum()}、"
                          f">100：{(age > 100).sum()}）")
    return df


def drop_out_of_bounds(df, removed):
    """步驟 4(c)：經緯度不在台澎金馬範圍（或空白）的案件，整起刪除"""
    lon = pd.to_numeric(df[COL_LONGITUDE], errors="coerce")
    lat = pd.to_numeric(df[COL_LATITUDE], errors="coerce")
    inside = pd.Series(False, index=df.index)
    for x0, x1, y0, y1 in GEO_BOXES.values():
        inside |= lon.between(x0, x1) & lat.between(y0, y1)
    return drop_cases(df, ~inside, removed, "經緯度異常",
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


def party_cols(col):
    """當事者欄位展開後的名稱：順位1_<col>、順位2_<col>"""
    return [p + col for p in PARTY_PREFIXES]


def resolve_cols(df, cols):
    """把欄位清單對應到 df 的實際欄位：案件欄位維持原名，當事者欄位換成 順位1_/順位2_ 兩欄"""
    out = []
    for c in cols:
        out += [c] if c in df.columns else [pc for pc in party_cols(c) if pc in df.columns]
    return out


def validate_cases(df):
    """步驟 5(b)：每起案件恰有順位 1、2，且順位前的案件層級欄位在同一案件內一致（不寫入輸出）"""
    grp = df.groupby(CASE_KEY, dropna=False, sort=False)
    ok = grp[COL_PARTY_ORDER].apply(lambda s: sorted(s.str.strip().astype(int)) == [1, 2])
    log(f"[驗證] 案件數 {len(ok)}，恰有順位 1、2 者 {ok.sum()}，異常 {(~ok).sum()}")
    case_cols = list(df.columns[: df.columns.get_loc(COL_PARTY_ORDER)])
    other = [c for c in case_cols if c not in CASE_KEY]
    incons = (grp[other].nunique(dropna=False) > 1)
    if incons.any().any():
        log(f"[驗證] 同一案件內其他案件層級欄位不一致：{incons.sum()[incons.sum() > 0].to_dict()}")
    else:
        log("[驗證] 同一案件內，當事者順位前的所有欄位皆一致")


def widen_parties(df):
    """步驟 5(c)：依當事者順位展開成一起案件一列，並移除當事者順位欄"""
    #--順位前的欄位為案件層級，順位後為當事者層級（原始檔的經度、緯度排在順位後，但屬案件鍵值）
    k = df.columns.get_loc(COL_PARTY_ORDER)
    after = list(df.columns[k + 1:])
    case_cols = list(df.columns[:k]) + [c for c in after if c in CASE_KEY]
    cols = [c for c in after if c not in CASE_KEY]
    order = df[COL_PARTY_ORDER].str.strip()
    p1 = df[order == "1"][case_cols + cols].rename(columns={c: f"順位1_{c}" for c in cols})
    p2 = df[order == "2"][CASE_KEY + cols].rename(columns={c: f"順位2_{c}" for c in cols})
    df = p1.merge(p2, on=CASE_KEY, how="left", validate="one_to_one")
    p2_cols = [f"順位2_{c}" for c in cols]
    log(f"[展開] 依當事者順位展開：{len(df)} 起案件（一起一列），順位2 缺漏 "
        f"{df[p2_cols].isna().all(axis=1).sum()} 起")
    return df


def to_case_rows(df):
    """步驟 5：只保留當事者順位 1、2，驗證後展開成一起案件一列"""
    keep = df[COL_PARTY_ORDER].str.strip().isin(["1", "2"])
    log(f"[篩選] 移除當事者順位 3 以後的資料 {(~keep).sum()} 列")
    df = df[keep].reset_index(drop=True)
    validate_cases(df)
    return widen_parties(df)


def add_location(df):
    """步驟 7(a)：由發生地點解析縣市、鄉鎮市區、高快速公路附近（已是一起案件一列）"""
    loc = parse_location(df)
    df = pd.concat([df, loc], axis=1)
    log(f"[地點] {len(loc)} 起案件；鄉鎮市區無法解析 {loc[COL_CITY_DISTRICT].isna().sum()} 筆；"
        f"縣市 {loc[COL_CITY].nunique()} 種、縣市鄉鎮市區 {loc[COL_CITY_DISTRICT].nunique()} 種；"
        f"高快速公路附近=Y {(loc[COL_IS_NEAR_HIGHWAY] == 'Y').sum()} 筆")
    return df


def drop_a1_overlap(df, removed):
    """步驟 3(b)：同一案件同時出現在 A1 與 A2 時，保留 A1"""
    k3 = [COL_INCIDENT_DATE, COL_INCIDENT_TIME, COL_ADDRESS]
    a1_keys = df.loc[df[COL_SEVERITY] == "A1", k3].drop_duplicates()
    in_a1 = df[k3].merge(a1_keys.assign(_a1=1), on=k3, how="left")["_a1"].notna().to_numpy()
    drop_mask = (df[COL_SEVERITY] == "A2").to_numpy() & in_a1
    log(f"[A1/A2] 刪除同時出現在 A1 的 A2 資料 {drop_mask.sum()} 列")
    return drop_rows(df, drop_mask, removed, "A1/A2重複（保留A1）")


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
    """步驟 7(d)：依發生時間建立時段（放在發生時間之後）"""
    hour = df[COL_INCIDENT_TIME].str[:2].astype(int)
    period = pd.Series(pd.NA, index=df.index, dtype="object")
    for h0, h1, name in PERIOD_BINS:
        period[(hour >= h0) & (hour < h1)] = name
    df.insert(df.columns.get_loc(COL_INCIDENT_TIME) + 1, COL_PERIOD, period)
    return df


def derive_columns(df):
    """步驟 7：衍生欄位（地點、死亡/受傷人數、當事者行動狀態、時段）"""
    df = add_location(df)
    df = split_casualties(df)
    df = merge_action(df)
    return add_period(df)


def correct_values(df):
    """步驟 8：資料內容校正（修正錯字、標記非人類當事者；順位1、2 分別處理），
    判定完成後去除 當事者屬-性-別名稱"""
    for wrong, right in TYPO_FIX.items():
        n_fix = 0
        for c in party_cols(COL_PROTECTION):
            n_fix += df[c].str.contains(wrong, na=False).sum()
            df[c] = df[c].str.replace(wrong, right, regex=False)
        log(f"[錯字] {COL_PROTECTION}：「{wrong}」→「{right}」{n_fix} 人")

    #--非人類當事者：年齡標 0；肇逃未查獲：年齡維持原值。兩者空白類別欄位各自補填標籤
    fill_party_blanks(df, NONHUMAN, NA_LABEL, "非人類當事者", zero_age=True)
    fill_party_blanks(df, UNSOLVED, UNSOLVED_LABEL, "肇逃未查獲當事者")
    return df.drop(columns=party_cols(COL_GENDER))


def fill_party_blanks(df, genders, label, name, zero_age=False):
    """當事者屬-性-別名稱 屬於 genders 者，空白的當事者類別欄位（PARTY_CAT_COLS）填入 label（順位1、2 分別處理）"""
    n_party = age_changed = n_fill = 0
    for p in PARTY_PREFIXES:
        hit = df[p + COL_GENDER].isin(genders)
        n_party += hit.sum()
        if zero_age:
            age_changed += (hit & (df[p + COL_AGE] != 0)).sum()
            df.loc[hit, p + COL_AGE] = 0
        for c in PARTY_CAT_COLS:
            m = hit & df[p + c].isna()
            n_fill += m.sum()
            df.loc[m, p + c] = label
    age_msg = f"年齡設為 0（其中原本非 0 者 {age_changed} 人），" if zero_age else ""
    log(f"[{name}] {n_party} 人：{age_msg}空白類別欄位填入「{label}」{n_fill} 格")


def convert_numeric(df):
    """步驟 6：數值欄位轉為數值型態（空字串 → 缺失）"""
    df = df.replace("", pd.NA)
    for c in resolve_cols(df, NUMERIC_COLS):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


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
        unknown = values[values.notna() & ~values.isin(mp)].unique()
        if len(unknown):
            log(f"[警告] {col} 有未定義對應的值，維持原值：{list(unknown)}")
        for c in cols:
            df[c] = df[c].map(lambda v: mp.get(v, v))
    log(f"[代碼] 已改為英文代碼：{'、'.join(CATEGORY_MAPS)}")

    #--時段（步驟 6(d) 建立）：依發生時間區間列入編碼對照
    hour = df[COL_INCIDENT_TIME].str[:2].astype(int)
    for h0, h1, name in PERIOD_BINS:
        in_bin = (hour >= h0) & (hour < h1)
        code_rows.append({"中文欄位": COL_PERIOD, "原始值": f"發生時間 {h0:02d}:00～{h1 - 1:02d}:59",
                          "英文代碼": name, "筆數": int(in_bin.sum())})
    return df, code_rows


def export(df, code_rows, out_path):
    """步驟 11：欄位改為英文名稱，輸出清洗結果，回傳 (欄位對照表, 編碼對照表)"""
    rename = dict(CASE_RENAME)
    for i in (1, 2):
        rename.update({f"順位{i}_{zh}": f"party{i}_{en}" for zh, en in PARTY_RENAME.items()})
    unmapped = [c for c in df.columns if c not in rename]
    if unmapped:
        raise ValueError(f"以下欄位沒有英文名稱：{unmapped}")
    dictionary = build_dictionary(df, rename)
    codebook = build_codebook(code_rows, rename)
    log(f"[欄位] 已改為英文名稱；編碼對照 {len(code_rows)} 列（併入總表）")
    df = df.rename(columns=rename)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"[輸出] {out_path}: {len(df)} 列 × {df.shape[1]} 欄")
    return dictionary, codebook


def clean_year(src_path, out_path):
    """清洗單一年度匯總檔 src_path，輸出到 out_path（附帶檔案放在同一資料夾），
    回傳該年度的 (欄位對照表, 編碼對照表, 輸出列數, 各移除原因列數)"""
    removed = []                                     # (移除原因, 原始列編號)
    raw = read_one(src_path)
    df = dedupe_cases(raw, out_path, removed)
    df = drop_bad_cases(df, removed)
    write_removed(raw, removed, removed_path(out_path))
    df = to_case_rows(df.drop(columns=RID))
    df = convert_numeric(df)
    df = derive_columns(df)
    df = correct_values(df)
    df = reorder_columns(df)
    df, code_rows = encode_categories(df)
    dictionary, codebook = export(df, code_rows, out_path)
    removed_counts = {}
    for reason, rids in removed:
        removed_counts[reason] = removed_counts.get(reason, 0) + len(rids)
    return dictionary, codebook, len(df), removed_counts


def write_summary(tables, path, label):
    """各年度表加上「年度」欄後合併，整份覆寫 path"""
    total = pd.concat([t.assign(年度=year)[["年度", *t.columns]] for year, t in tables],
                      ignore_index=True)
    total.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"[總表] {label} {path}: {len(total)} 列（{len(tables)} 個年度）")


def read_summary(path):
    """讀回上次輸出的總表（全部以字串讀入，原樣寫回）；不存在時回傳 None"""
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def summary_rows(total, year):
    """取出總表中該年度的列（去掉「年度」欄）；總表不存在或沒有該年度時回傳 None"""
    if total is None:
        return None
    rows = total[total["年度"] == year]
    return rows.drop(columns="年度").reset_index(drop=True) if len(rows) else None


def source_sha256(src, merged_record):
    """來源檔 sha256：merger.py 紀錄的輸出檔名相符時直接沿用，否則重新計算"""
    info = merged_record.get(f"Y{src.stem.rsplit('_Y', 1)[-1]}", {})
    if info.get("output_file") == src.name and info.get("output_sha256"):
        return info["output_sha256"]
    return sha256_of_file(src)


def clean_all():
    """逐年清洗 data/processed/merged_years/ 底下的年度匯總檔，回傳本次有重新清洗的輸出檔路徑清單。
    來源檔與清洗規則檔皆未變更、輸出檔與總表中該年度的列都在時，略過該年度。"""
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    sources = sorted(MERGED_YEARS_DIR.glob("TW_traffic_accident_Y*.csv"))
    if not sources:
        log(f"[skip] {MERGED_YEARS_DIR} 找不到年度匯總檔，請先執行 merger.py")
        return []

    record = load_json_record(CLEANED_RECORD_PATH)
    merged_record = load_json_record(MERGED_RECORD_PATH)
    rule_hashes = {p.name: sha256_of_file(p) for p in RULE_FILES}
    old_dicts, old_codebooks = read_summary(DICT_PATH), read_summary(CODEBOOK_PATH)

    outputs, dicts, codebooks = [], [], []
    for src in sources:
        year = src.stem.rsplit("_Y", 1)[-1]
        out_path = CLEANED_DIR / f"{src.stem}_cleaned.csv"
        src_hash = source_sha256(src, merged_record)

        #--檢查是否可略過：來源、規則皆未變更，且輸出檔、移除資料檔、總表中該年度的列都在
        previous = record.get(f"Y{year}")
        dictionary, codebook = summary_rows(old_dicts, year), summary_rows(old_codebooks, year)
        if (
            previous is not None
            and previous.get("source_sha256") == src_hash
            and previous.get("rule_files") == rule_hashes
            and out_path.exists()
            and removed_path(out_path).exists()
            and dictionary is not None
            and codebook is not None
        ):
            log(f"[skip] {src.name}：來源檔與清洗規則未變更，略過重新清洗")
        else:
            log(f"===== {src.name} -> {out_path.name} =====")
            dictionary, codebook, row_count, removed_counts = clean_year(src, out_path)
            record[f"Y{year}"] = {
                "source_file": src.name,
                "source_sha256": src_hash,
                "rule_files": rule_hashes,
                "output_file": out_path.name,
                "output_row_count": row_count,
                "output_sha256": sha256_of_file(out_path),
                "removed_file": removed_path(out_path).name,
                "removed_counts": removed_counts,
                "cleaned_at": datetime.now(timezone.utc).isoformat(),
            }
            outputs.append(out_path)
        dicts.append((year, dictionary))
        codebooks.append((year, codebook))

    #--只有在真的有清洗到任何年度時，才重新產生總表、寫回 registry
    if outputs:
        write_summary(dicts, DICT_PATH, "欄位對照表")
        write_summary(codebooks, CODEBOOK_PATH, "編碼對照表")
        save_json_record(CLEANED_RECORD_PATH, record)
    return outputs


if __name__ == "__main__":
    clean_all()
