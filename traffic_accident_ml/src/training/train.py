"""
模型訓練流程(train.py)
- 目標：severity(A1 為 1、A2 為 0)，以邏輯斯迴歸估計勝算比
- 流程：
    1. 讀取資料：各年度清洗後資料集，只保留案件層級欄位，並移除 LOW_MI_COLS
    2. 切分資料：依 year × severity 分層切成 train / validate / test(0.7 / 0.2 / 0.1)
       - 訓練集再欠採樣成多個子集(A1:A2 = 1:SAMPLE_FOLD)，先只用第 SUBSET_INDEX 個
       - validate、test 不欠採樣，維持原始比例
    3. 訓練模型：One-Hot 編碼規則以完整訓練集決定，邏輯斯迴歸以欠採樣子集配適
    4. 評估模型：只在 validate 計算 PR-AUC、recall、precision(切點 0.5)，作為比較、挑選版本的依據
       - test 不在訓練時使用，避免調參時偷看；版本定案後以 evaluate.py 評估一次
    5. 輸出結果：模型(.joblib)、勝算比表(.csv)存至 models/，訓練紀錄附加至 models/metadata.json

執行：python -m src.training.train
"""

from datetime import datetime, timezone

import joblib
import sklearn

from src.data_management.common import BASE_DIR, load_json_record, save_json_record
from src.preprocessing.encoding import MIN_FREQUENCY, fitted_reference_levels
from src.preprocessing.loader import (
    DEFAULT_YEARS, LOW_MI_COLS, RANDOM_STATE, SPLIT_RATIO_TRAIN_TO_TEST,
    get_ds_with_adjusted_cols, split_with_undersampling,
)
from src.preprocessing.target import build_xy
from src.training import severity_model
from src.training.evaluate import severity_metrics

#--模型、勝算比表、訓練紀錄的輸出位置
MODELS_DIR = BASE_DIR / "models"
METADATA_PATH = MODELS_DIR / "metadata.json"
#--欠採樣：每個訓練子集 A1:A2 = 1:SAMPLE_FOLD，訓練時使用第 SUBSET_INDEX 個子集
SAMPLE_FOLD = 5
SUBSET_INDEX = 0


def log(msg):
    """印出帶時間的訊息，方便觀察各步驟耗時"""
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def base_record(target, stamp, model, X_train, n_rows, model_path, table_path, sampling):
    """
    建立一筆訓練紀錄的共通欄位，寫入 metadata.json 作為比較不同版本模型的依據
    - 資料：年度、切分比例、亂數種子、欠採樣設定、移除的欄位、各子集筆數
    - 編碼：使用的特徵、參考類別、併入 infrequent 的門檻
    - 產出：模型檔、係數表檔名
    - 採用哪一個版本不記在這裡，由 promote.py 登錄於 registry/manifest.json
    """
    return {
        "timestamp": stamp,
        "target": target,
        "years": list(DEFAULT_YEARS),
        "split_ratios": list(SPLIT_RATIO_TRAIN_TO_TEST),
        "random_state": RANDOM_STATE,
        "sampling": sampling,
        "dropped_cols": LOW_MI_COLS,
        "rows": n_rows,
        "features": list(X_train.columns),
        "reference_levels": fitted_reference_levels(model),
        "min_frequency": MIN_FREQUENCY,
        "model_file": model_path.name,
        "coef_table_file": table_path.name,
    }


def train_severity(splits, stamp, sampling):
    """
    訓練模型、評估模型、輸出結果 
    - severity 配適邏輯斯迴歸
    - 回傳 (訓練紀錄, 勝算比表)
    """
    #--設定訓練集/訓練子集/驗證集的特徵 X 與目標 y
    (X_encode, _), (X_tr, y_tr), (X_va, y_va) = [build_xy(d) for d in splits]

    #-- 3.訓練模型：3-1.One-Hot 編碼。3-2.fit 邏輯斯迴歸
    #--model 是一個 sklearn Pipeline，包括 "prepare", "onehot", "drop_rare", "clf"
    log("[severity_training] 配適邏輯斯迴歸")
    model = severity_model.fit_severity(X_encode, X_tr, y_tr)
    #--檢查訓練的收斂情形：n_iter_ 為實際迭代次數，達到 max_iter 上限表示可能沒收斂
    clf = model.named_steps["clf"]
    n_iter = int(clf.n_iter_[0])
    converged = n_iter < clf.max_iter
    log(f"[severity_training] 迭代 {n_iter}/{clf.max_iter} 次，{'已收斂' if converged else '未收斂，係數可能不準確'}")
    
    #-- 4.評估模型：predict_proba 的第 2 欄為 A1 的預測機率
    metrics = {
        "validate": severity_metrics(y_va, model.predict_proba(X_va)[:, 1]),
    }
    #--驗證集指標：PR-AUC 與亂猜(≈ A1 占比)比較
    v = metrics["validate"]
    log(f"[severity_evaluating] validate PR-AUC={v['average_precision']:.4f}(亂猜≈{v['baseline_average_precision']:.4f})，"
        f"recall={v['recall']:.3f}，precision={v['precision']:.3f}")

    #-- 5. 輸出結果
    table = severity_model.odds_ratio_table(model, X_tr, y_tr)
    #--產生模型檔(.joblib)：整個 Pipeline 建模，用於 evaluate.py 在測試集評估、predictor.py 對外提供預測
    #--產生勝算比表(.csv)：每列為一個 One-Hot 類別的係數與勝算比，供人工解讀各因素對 A1 的影響    
    model_path = MODELS_DIR / f"severity_{stamp}.joblib"
    table_path = MODELS_DIR / f"severity_{stamp}_odds_ratio.csv"
    joblib.dump(model, model_path)
    table.to_csv(table_path, index=False, encoding="utf-8-sig")

    #--產生訓練紀錄
    record = base_record("severity", stamp, model, X_tr, 
                         {"train": len(X_tr), "validate": len(X_va)},
                         model_path, table_path, sampling)
    record.update({
        "estimator": type(clf).__name__,
        "estimator_params": clf.get_params(),   #--記錄本次訓練實際使用的全部超參數(含預設值)
        "sklearn_version": sklearn.__version__,
        "positive_label": "A1",
        "n_iter": n_iter,
        "converged": converged,
        "metrics": metrics,
    })
    return record, table


def append_metadata(records):
    """把本次訓練紀錄附加到 metadata.json 既有紀錄之後"""
    history = load_json_record(METADATA_PATH) or []
    save_json_record(METADATA_PATH, history + records)


def load_splits():
    """
    讀取資料、切分資料
    - 回傳 (splits, sampling)
        - splits：(完整訓練集, 訓練子集, 驗證集)
        - sampling：欠採樣設定，寫入訓練紀錄
    """
    #-- 1.讀取資料：只保留案件層級欄位
    log(f"讀取資料 train/validate，訓練集欠採樣 A1:A2 = 1:{SAMPLE_FOLD}")
    df = get_ds_with_adjusted_cols()

    #-- 2.切分資料：欠採樣後的訓練子集，以及維持原始比例的驗證集；測試集留給 evaluate.py，這裡不使用
    train_full, train_subsets, validate, _ = split_with_undersampling(
        df, sample_fold=SAMPLE_FOLD, drop_cols=LOW_MI_COLS)
    #--編碼規則：以完整訓練集決定；模型訓練：以欠採樣單一子集訓練
    splits = (train_full, train_subsets[SUBSET_INDEX], validate)
    #--整理訓練紀錄：欠採樣設定
    sampling = {"method": "undersampling", "sample_fold": SAMPLE_FOLD,
                "n_subsets": len(train_subsets), "subset_index": SUBSET_INDEX}
    log(f"共 {len(train_subsets)} 個訓練子集，使用第 {SUBSET_INDEX} 個：{len(splits[1])} 筆")
    return splits, sampling


def main():
    #--建立時戳，用於當次訓練的所有檔名
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    #-- 1.2.讀取並切分資料
    splits, sampling = load_splits()
    #-- 3.4.5.訓練、評估並印出勝算比前 20 名
    record, table = train_severity(splits, stamp, sampling)
    print(table.head(20).to_string())

    #--寫入訓練紀錄
    append_metadata([record])
    log(f"完成，模型與係數表已輸出至 {MODELS_DIR}，紀錄已寫入 {METADATA_PATH.name}")


if __name__ == "__main__":
    main()
