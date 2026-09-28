"""
目標與特徵定義(target.py)
- 定義目標 y 與不列入特徵 X 的欄位
- 同時提供：特徵篩選(filter_selection.py)、訓練(train.py)
"""

from src.data_management.cleaning_rules import (
    CASE_RENAME,
    COL_SEVERITY, COL_YEAR, COL_DEAD_NUM, COL_INJURED_NUM,
)

#--目標欄位、二元分類的正類依據
#--A1車禍：車禍當下或車禍發生24小時內有人死亡
#--A2車禍：車禍有人員受傷或車禍發生24小時後有人死亡
#--A3車禍：僅有車輛或財物受損的車禍
TARGET_COL = CASE_RENAME[COL_SEVERITY]
POSITIVE_LABEL = "A1"

#--不列入特徵的欄位
#--分割X使用：年度、嚴重度
#--目標y設定：嚴重度、死亡人數、受傷人數
EXCLUDE_COLS = [CASE_RENAME[c] for c in (COL_SEVERITY, COL_YEAR, COL_DEAD_NUM, COL_INJURED_NUM)]


def build_xy(df):
    """ 
    拆出特徵 X 與目標 y
    """
    #--指定X欄位(特徵)
    X = df.drop(columns=EXCLUDE_COLS)
    #--指定y欄位(A1當作1，其餘如A2都是0)
    y = (df[TARGET_COL] == POSITIVE_LABEL).astype(int)
    return X, y
