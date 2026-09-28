"""
模型訓練(severity_model.py)
- Logistic Regression, 分類模型(二元分類)
- exp(係數) = 勝算比(odds ratio)
- 編碼：訓練集完整樣本
- 訓練：以欠採樣調整為 A1:A2 = 1:5 (class_weight="balanced" 暫不使用)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing.encoding import build_preprocessor_steps, fitted_reference_levels, split_feature_name


def fit_severity(X_encode, X, y):
    """
    訓練：One-Hot 編碼 Logistic 迴歸
    - X_encode：決定 One-Hot 編碼規則的資料(完整訓練集)
    - X, y：訓練模型的資料(欠採樣子集)
    原本寫法：model = Pipeline([prepare, onehot, drop_rare, clf]).fit(X, y)
    為符合編碼用完整訓練集、模型用子集，才採用分開拆成兩個 Pipeline。
    """
    #-- 1.建立前處理器：學習編碼原則
    #-- 每個欄位有哪些類別、哪些類別要併入 infrequent、哪些 One-Hot 欄位要保留
    preprocessor = Pipeline(build_preprocessor_steps(X_encode)).fit(X_encode)
    #-- 2.建立分類器：取得學好的編碼規則轉換子集，再訓練模型
    #-- ConvergenceWarning 是跑滿 max_iter 上限仍無法收斂的情形，常發生於資料量大(欄位多)或類別不平衡
    # clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf = LogisticRegression(max_iter=1000).fit(preprocessor.transform(X), y) #--訓練集改用欠採樣本比例 A1:A2，不另外加權
    #-- 3.組成完整模型
    #-- One-Hot 規則要用完整訓練集決定，而分類器用欠採樣子集配適
    return Pipeline(preprocessor.steps + [("clf", clf)])


def odds_ratio_table(model, X, y):
    """ 
    勝算比表
    - 每列為一個 One-Hot 類別(參考類別不列出) 
    """
    #--轉成 numpy 陣列，才能當成布林遮罩篩選 Z 的列
    y = np.asarray(y)
    #--Pipeline 最終估計器之前的輸出(已編碼的設計矩陣)
    #-- model 是 prepare → onehot → drop_rare → clf
    #-- model[:-1] 為 Pipeline 的前半段，只完成到編碼
    #-- model[-1] 是 Pipeline 的後半段，只有模型本身
    #-- X (編碼前) -> Z (編碼後)：列數不變，但欄數變多
    Z = model[:-1].transform(X) #--type: csr_matrix
    #--從模型取出「欄位=類別」，所有 Z 各欄的欄名，順序與係數一一對應
    names = model[:-1].get_feature_names_out()
    #--取出訓練前的欄位參考類別(通常是對常出現的類別)
    refs = fitted_reference_levels(model)

    #--建立勝算比表
    #--每個 One-Hot 欄位一列，欄名被拆成：feature(欄位)、level(類別)
    table = pd.DataFrame([split_feature_name(n) for n in names], 
                         columns=["feature", "level"], index=names)
    #--找出參考類別比較
    table["reference"] = table["feature"].map(refs)
    #--該類別的所有案件數：Z 每欄加總，該欄是 1 的總列數
    #--該類別是 A1 的案件數：限 y == 1 的列數
    table["n"] = np.asarray(Z.sum(axis=0)).ravel().astype(int)
    table["n_a1"] = np.asarray(Z[y == 1].sum(axis=0)).ravel().astype(int)
    #--取出係數：正值是比參考類別更容易是 A1，負值是較不容易
    #--coef_ 形狀為 (1, 欄數)，ravel() 攤平成一維才能放進表格
    table["coef"] = model.named_steps["clf"].coef_.ravel()
    #--勝算比 = exp(係數)
    #-- > 1 表示 A1 的勝算是參考類別的幾倍，< 1 表示較低，= 1 表示無差異
    table["odds_ratio"] = np.exp(table["coef"])
    #--依勝算比由大到小排序，把最容易造成 A1 的類別正序排序
    return table.sort_values("odds_ratio", ascending=False).reset_index(drop=True)
