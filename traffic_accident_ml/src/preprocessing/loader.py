"""
資料載入(loader.py)
讀取各年度清洗後的事故資料 csv，
依 cleaning_rules 篩選欄位，並依年度 × 嚴重度分層切分 train/validate/test。

"""

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
SPLIT_RATIO_TRAIN_TO_TEST = (0.7, 0.2, 0.1)  #--訓練、驗證、測試
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

def split_by_year_severity(df, ratio_train_to_test=SPLIT_RATIO_TRAIN_TO_TEST, random_state=RANDOM_STATE, drop_cols=None):
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
        n_train = round(len(g) * ratio_train_to_test[0])
        n_validate = round(len(g) * ratio_train_to_test[1])
        train.append(g.iloc[:n_train])
        validate.append(g.iloc[n_train:n_train + n_validate])
        test.append(g.iloc[n_train + n_validate:])
    return tuple(pd.concat(parts).drop(columns=drop_cols) for parts in (train, validate, test))

def split_with_undersampling(df, sample_fold=5, ratio_train_to_test=SPLIT_RATIO_TRAIN_TO_TEST, random_state=RANDOM_STATE, drop_cols=None):
    """
    欠採樣(undersampling)
    - 將 split_by_year_severity 的訓練集細分成多個子集，A1:A2 = 1:sample_fold。
    - 每個訓練子集資料數 = 全部 A1 + a2_subset_max 筆 A2。A2 不分年度打亂後依序切段，子集之間不重複。
    - 只調整訓練子集的資料，驗證集、測試集維持原始樣本分配。
    - 回傳 (完整訓練集, 訓練子集 list, 驗證集, 測試集)；完整訓練集可用來決定 One-Hot 編碼規則
    - 可能延伸：集成學習(ensemble learning)
    """
    #--取出已分好訓練集/驗證集/測試集
    train, validate, test = split_by_year_severity(df, ratio_train_to_test, random_state, drop_cols)
    #--重新打亂訓練集的A2資料，再分配到訓練子集(subset)
    #--最後一組訓練子集的A2，若無法湊滿，將不使用。
    is_a1 = train[CASE_RENAME[COL_SEVERITY]] == "A1"
    a1 = train[is_a1]
    a2 = train[~is_a1].sample(frac=1, random_state=random_state)
    a2_subset_max = len(a1) * sample_fold
    train_subsets = [pd.concat([a1, a2.iloc[i:i + a2_subset_max]])
                     for i in range(0, len(a2) - a2_subset_max + 1, a2_subset_max)]
    return train, train_subsets, validate, test

if __name__ == "__main__":
    df_ori = get_ds_with_adjusted_cols()
    # df_train, df_validate, df_test = split_by_year_severity(df_ori, drop_cols=None)
    # df_train, df_validate, df_test = split_by_year_severity(df_ori, drop_cols=LOW_MI_COLS)
    df_train_full, train_subsets, df_validate, df_test = split_with_undersampling(df_ori, sample_fold=3, drop_cols=LOW_MI_COLS)
    train_subsets[0].info()
    print(len(train_subsets))
    #--檢查各子集的年度 × 事故類別筆數
    for name, d in (("train", train_subsets[0]), ("validate", df_validate), ("test", df_test)):
        print(f"--- {name}: {len(d)} 筆")
        #--顯示樞紐表：年別 × 嚴重度
        print(pd.crosstab(d[CASE_RENAME[COL_YEAR]], d[CASE_RENAME[COL_SEVERITY]]))