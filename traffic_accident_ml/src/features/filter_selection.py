"""
過濾法欄位篩選
- 各欄位單獨與目標 y（是否為 A1）計算互資訊，不需訓練模型，快速檢視關聯強度。
- 捕捉非線性關係，分類和回歸都適用：mutual_info_regression、mutual_info_classif
- H(y)、MI、MI%、p-value。
- p-value < 0.01 可保留，對y不是雜訊。移除 rd_defect, rd_obstacle, rd_surface, rd_slippery
"""

import pandas as pd
from scipy.stats import chi2, entropy
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

from src.preprocessing.loader import get_ds_with_adjusted_cols, split_by_year_severity
from src.preprocessing.target import POSITIVE_LABEL, TARGET_COL, build_xy


def mutual_info_ranking(X, y):
    """
    計算各欄位與 y 的互資訊，依 mi 由大到小排序
    - 全部欄位視為類別(discrete_features=True)，編碼代碼的順序不影響結果
    - mi_ratio_%：mi 占 y 本身資訊量 H(y) 的比例；A1 占比極低使 H(y) 很小，宜看相對大小
    - g_pval：G 檢定 p 值，G = 2 × N × MI 近似卡方分布，p 值大表示與 y 的關聯可能只是雜訊
      資料量大時幾乎都會顯著，只能用來排除雜訊，不代表欄位重要
    """
    #--mutual_info_classif 只接受數值，先轉字串再編碼，避免數值與文字混型
    X_encoded = OrdinalEncoder().fit_transform(X.astype(str))
    #--計算每欄與 y 的互資訊，並以欄名作為索引
    mi = pd.Series(mutual_info_classif(X_encoded, y, discrete_features=True), index=X.columns)
    #--y 本身的資訊量(熵)，作為互資訊的上限與比例分母
    h_y = entropy(y.value_counts(normalize=True))
    #--G 檢定：mi 為自然對數單位(nat)，G = 2 × N × MI；自由度 = (欄位類別數 − 1) × (y 類別數 − 1)
    g_stat = 2 * len(y) * mi
    dof = (X.nunique() - 1) * (y.nunique() - 1)
    #--卡方分布右尾機率即 p 值
    g_pval = pd.Series(chi2.sf(g_stat, dof), index=X.columns)
    #--彙整互資訊、占比與檢定結果，由大到小排序
    result = pd.DataFrame({"mi": mi, "mi_ratio_%": mi / h_y * 100,
                           "g_stat": g_stat, "dof": dof, "g_pval": g_pval})
    return result.sort_values("mi", ascending=False)


def category_positive_rate(df, col):
    """各類別的筆數與 A1 比例，補足互資訊看不出是哪個類別造成差異的部分"""
    is_positive = df[TARGET_COL] == POSITIVE_LABEL
    result = is_positive.groupby(df[col]).agg(count="size", a1_rate="mean")
    return result.sort_values("a1_rate", ascending=False)


if __name__ == "__main__":
    #--以severity為y，取出訓練集建立X和y
    df_train, _, _ = split_by_year_severity(get_ds_with_adjusted_cols())
    X, y = build_xy(df_train)
    ranking = mutual_info_ranking(X, y)

    pd.set_option("display.float_format", "{:.5f}".format)
    #--H(y)表示未知任何欄位時，猜中y的難度
    print(f"H(y) = {entropy(y.value_counts(normalize=True)):.5f}")
    #--MI表示知道特定一欄，猜中y的難度。
    #--MI越大，越容易猜中A1。
    #--MI% = MI / H(y)，單欄能解釋猜中貢獻的最大比例。
    #--前六名是acc_subtype, acc_type, rd_type, period, rd_incident_sub, city
    print(ranking)
    #--只先取排優先順序第1的acc_subtype來看
    top_col = ranking.index[0]
    print(f"\n--- {top_col} 各類別 A1 比例")
    print(category_positive_rate(df_train, top_col))
