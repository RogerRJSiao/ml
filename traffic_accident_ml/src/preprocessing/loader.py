"""讀取多年度事故資料 csv，依 schema.py 驗證欄位，合併成單一 DataFrame。"""

import pandas as pd

from src.data_management.common import BASE_DIR

from src.data_management.cleaning_rules import (
    CASE_RENAME,
    COL_INCIDENT_DATE, COL_INCIDENT_TIME,
    COL_LONGITUDE, COL_LATITUDE, COL_POLICE, COL_ADDRESS, COL_CITY_DISTRICT,
    #--拆分子集的分組依據
    COL_YEAR, COL_SEVERITY,
    #--特徵篩選MI，與severity較無相關的
    COL_RD_SURFACE, COL_RD_SLIPPERY, COL_RD_DEFECT, COL_RD_OBSTACLE,
)

#--清洗後資料集目錄（cleaner.py 的輸出）
CLEANED_DIR = BASE_DIR / "data" / "processed" / "cleaned"
#--預設載入的西元年度
DEFAULT_YEARS = (2020, 2021, 2022)
#--資料子集分配比例、亂數種子
SPLIT_RATIOS = (0.7, 0.2, 0.1)  #--訓練、驗證、測試
RANDOM_STATE = 42               #--固定亂數種子，每次執行切出來的結果都一樣
#--過濾法(filter_selection.py)判定與 y 無顯著關聯的欄位：互資訊落在雜訊範圍、G 檢定 p 值 > 0.01
LOW_MI_COLS = [CASE_RENAME[c] for c in (COL_RD_SURFACE, COL_RD_SLIPPERY, COL_RD_DEFECT, COL_RD_OBSTACLE)]


def get_cleaned_path(year):
    return CLEANED_DIR / f"TW_traffic_accident_Y{year}_cleaned.csv"


def load_all_years(years=DEFAULT_YEARS):
    """讀取指定 cleaned.csv，讀出全部欄位，依年度順序上下合併成單一 DataFrame"""
    #--先確認檔案都在，缺檔直接報錯，避免讀到一半才失敗
    paths = [get_cleaned_path(year) for year in years]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"找不到清洗後資料集：{missing}，請先執行 cleaner.py")

    #--cleaner.py 已將數值欄位轉型、空白以缺失值表示，這裡交由 pandas 推斷型別
    #--low_memory=False：整檔一次推斷型別，避免大檔分塊讀取造成同欄混型
    frames = [pd.read_csv(p, encoding="utf-8-sig", low_memory=False) for p in paths]
    #--pd.read_csv 預設 header=0，故不會有header寫入
    return pd.concat(frames, ignore_index=True)

def get_ds_with_adjusted_cols():
    """
    調整資料集，預備拆分成子集使用
    - 移除PARTY欄位、流水帳資料欄位
    - 樣本分配前，不可移除 year, severity
    """
    df = load_all_years()
    #--只保留 CASE_RENAME 登錄的案件層級欄位，依 csv 原欄序排列
    case_cols = set(CASE_RENAME.values())
    df = df[[c for c in df.columns if c in case_cols]]
    #--移除流水帳資料或不分析欄位
    unused_cols_ch = [COL_INCIDENT_DATE, COL_INCIDENT_TIME, 
                      COL_LONGITUDE, COL_LATITUDE, COL_POLICE, COL_ADDRESS, COL_CITY_DISTRICT]
    unused_cols_en = [CASE_RENAME[c] for c in unused_cols_ch]
    df = df.drop(columns=unused_cols_en)

    #--查看欄位摘要或資料分布(數值/文字、空值、唯一值)
    # print(df.describe(include="all").T)
    # df.info()
    return df

def split_by_year_severity(df, ratios=SPLIT_RATIOS, random_state=RANDOM_STATE, drop_cols=None):
    """
    拆分資料集
    - 依 year × severity 分組，各組各自打亂後按比例切成訓練、驗證、測試子集，再合併
    - 各子集的年度與事故類別比例與原資料一致，A1 樣本少也不會集中在某一子集
    - 隨機種子為固定值，利用偽隨機(pseudo-random)特性，讓每次抽樣維持一致(=再現性)。
    - drop_cols：切分後自各子集移除的欄位(例如 LOW_MI_COLS)；不影響分組結果，同一種子下列的分配與不移除時相同
    """
    col_year, col_severity = CASE_RENAME[COL_YEAR], CASE_RENAME[COL_SEVERITY]
    drop_cols = list(drop_cols or [])
    #--year、severity 為分組依據，不可移除
    is_protected = {col_year, col_severity} & set(drop_cols)    #--對欄名取交集
    if is_protected:
        raise ValueError(f"drop_cols 不可包含分組欄位：{sorted(is_protected)}")
    
    #--根據年度yyyy、嚴重度A1/A2分組，分組分配樣本
    train, validate, test = [], [], []
    for _, g in df.groupby([col_year, col_severity]):
        #--取出 100% 資料比數，打亂順序，減少日期排序影響結果
        g = g.sample(frac=1, random_state=random_state)
        #--訓練集先拿 0.7、驗證集再拿 0.2，最後剩餘的都歸測試集(原定0.1)
        n_train = round(len(g) * ratios[0])
        n_validate = round(len(g) * ratios[1])
        train.append(g.iloc[:n_train])
        validate.append(g.iloc[n_train:n_train + n_validate])
        test.append(g.iloc[n_train + n_validate:])
    return tuple(pd.concat(parts).drop(columns=drop_cols) for parts in (train, validate, test))

if __name__ == "__main__":
    df_ori = get_ds_with_adjusted_cols()
    # df_train, df_validate, df_test = split_by_year_severity(df_ori, drop_cols=None)
    df_train, df_validate, df_test = split_by_year_severity(df_ori, drop_cols=LOW_MI_COLS)
    # df_test.info()
    #--檢查各子集的年度 × 事故類別筆數
    for name, d in (("train", df_train), ("validate", df_validate), ("test", df_test)):
        print(f"--- {name}: {len(d)} 筆")
        print(pd.crosstab(d[CASE_RENAME[COL_YEAR]], d[CASE_RENAME[COL_SEVERITY]]))