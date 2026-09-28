"""
類別欄位 One-Hot 編碼
- Pipeline 前半段
- 全部欄位都是類別，只做 One-Hot，不需要 StandardScaler
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


#--排除共線性
#--rd_signals_defect 的 no_signals 與 rd_signals 的 no_signals 是同一批資料
#--rd_signals_defect 只保留「號誌是否故障」的資訊：無號誌、正常 → not_malfunction
SIGNALS_DEFECT_RECODE = {"no_signals": "not_malfunction", "normal": "not_malfunction"}

#--定義參考類別(同一欄位內資料內容的參考類別)
#--把最常見的資料內容當勝算比的基準，係數解讀為「該類別相對於基準」
REFERENCE_LEVELS = {
    "weather": "sunny",
    "light": "natural",
    "period": "daytime",
    "rd_type": "urban",
    "rd_vision": "good",
    "rd_signals": "signals",
    "rd_signals_defect": "not_malfunction",
    "rd_shoulder": "Y",
    "is_near_highway": "N",
    "acc_type": "two_vehicles",
}

#--筆數門檻(以 fit 編碼的完整訓練集計算)，用於兩處：
#--1. OneHotEncoder：筆數 < 門檻的類別，合併成 infrequent(參考類別除外)
#--2. DropRareColumns：合併後欄位筆數仍 < 門檻，整欄移除，歸入參考類別
MIN_FREQUENCY = 200

#--get_feature_names_out 的欄名分隔符：欄位=類別
NAME_DELIMITER = "="


def prepare_X(X):
    """全部欄位轉字串，並合併共線的類別"""
    #--全部欄位資料轉成文字
    X = X.astype(str)

    #--處理共線性的欄位資料
    if "rd_signals_defect" in X.columns:
        X["rd_signals_defect"] = X["rd_signals_defect"].replace(SIGNALS_DEFECT_RECODE)
    return X


def _combine_name(feature, category):
    #--OneHotEncoder 輸出欄名
    #--需為模組層級函式，模型才能以 joblib 存檔
    return f"{feature}{NAME_DELIMITER}{category}"


def build_one_hot(X, min_frequency=MIN_FREQUENCY):
    """ 依訓練集X(需先經過 prepare_X)建立 OneHotEncoder，drop 各欄參考類別 """
    #--各欄參考類別：有指定就用指定值，否則取訓練集最多筆的類別
    refs = [REFERENCE_LEVELS.get(c, X[c].mode()[0]) for c in X.columns]
    #--執行 one hot 編碼
    #--移除參考類別欄名，遇到沒看過的類別，輸出稀疏矩陣，輸出欄名格式
    #--handle_unknown='ignore'：推論時沒看過的類別全為 0，等同參考類別
    return OneHotEncoder(drop=refs, handle_unknown="ignore",
                         min_frequency=min_frequency, sparse_output=True,
                         feature_name_combiner=_combine_name)


class DropRareColumns(TransformerMixin, BaseEstimator):
    """
    移除訓練集筆數 < min_frequency(預設 MIN_FREQUENCY)的 One-Hot 欄位，這些列歸入參考類別
    - 筆數太少的欄位係數估不準，也容易與其他欄位近似共線
    """

    def __init__(self, min_frequency=MIN_FREQUENCY):
        self.min_frequency = min_frequency

    def fit(self, X, y=None):
        #--各欄加總 = 該類別在訓練集的筆數；記下哪些欄位筆數足夠要保留
        self.keep_ = np.asarray(X.sum(axis=0)).ravel() >= self.min_frequency
        return self

    def transform(self, X):
        #--只取保留的欄位；被移除的欄位不再出現，這些列等同參考類別
        return X[:, np.flatnonzero(self.keep_)]

    def get_feature_names_out(self, input_features=None):
        #--輸出欄名格式：欄名跟著同步篩選，讓係數表對齊
        return np.asarray(input_features)[self.keep_]


def build_preprocessor_steps(X_train):
    """ 
    Pipeline 前半段
    step 1. prepare_X：轉成文字、合併共線類別
    step 2. One-Hot：文字轉成 0/1 欄位，移除參考類別
    step 3. 移除筆數太少的欄位，避免雜訊
    - 後半段接續：設計矩陣 -> 交給模型
    """
    #--三個步驟的名稱、轉換器
    return [("prepare", FunctionTransformer(prepare_X, feature_names_out="one-to-one")),
            ("onehot", build_one_hot(prepare_X(X_train))),
            ("drop_rare", DropRareColumns())]


def fitted_reference_levels(model):
    """ 從配適好的 Pipeline 讀出各欄實際使用的參考類別，報表與模型一定一致 """
    enc = model.named_steps["onehot"]
    return {c: str(cats[i]) for c, cats, i in zip(enc.feature_names_in_, enc.categories_, enc.drop_idx_)}


def split_feature_name(name):
    """ rd_signals=no_signals' → ('rd_signals', 'no_signals') """
    feature, level = name.split(NAME_DELIMITER, 1)
    return feature, level
