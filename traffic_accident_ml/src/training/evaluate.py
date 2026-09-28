"""
評估指標(evaluate.py)
- 不使用 accuracy：全部猜 A2 就有 99.5% 準確率
- severity：PR-AUC(average_precision_score)、recall、precision、混淆矩陣；對照基準為 A1 占比
- 測試集評估：版本定案後才以此入口在 test 評估一次，結果寫回該筆訓練紀錄
    - 每筆紀錄只評估一次，已有 test 指標時拒絕執行，避免反覆看 test 分數調參
    - 依紀錄中的年度、切分比例、亂數種子、移除欄位重建相同的測試集

執行：python -m src.training.evaluate <訓練紀錄 timestamp>
     例如 python -m src.training.evaluate 20260928T081841Z
"""

import argparse
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix,
    precision_score, recall_score, roc_auc_score,
)

from src.data_management.common import load_json_record, save_json_record
from src.preprocessing.loader import DEFAULT_YEARS, get_ds_with_adjusted_cols, split_by_year_severity
from src.preprocessing.target import build_xy

#--預測機率 ≥ 此值判為 A1
THRESHOLD = 0.5


def severity_metrics(y, proba, threshold=THRESHOLD):
    y = np.asarray(y)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, proba)),
        #--隨機亂猜的 PR-AUC ≈ 正類占比
        "baseline_average_precision": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, proba)),
        "threshold": threshold,
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate_test(timestamp, force=False):
    """
    在測試集評估指定的訓練紀錄，並把 test 指標寫回 metadata.json
    - timestamp：訓練紀錄的 timestamp(也是模型檔名的時戳)
    - force：已有 test 指標時仍重新評估；只在確定需要時使用
    """
    #--train.py 會 import 本模組的 severity_metrics，放在函式內 import 避免循環匯入
    from src.training.train import METADATA_PATH, MODELS_DIR

    #--找出指定的訓練紀錄
    history = load_json_record(METADATA_PATH) or []
    matches = [r for r in history if r["timestamp"] == timestamp]
    if not matches:
        raise ValueError(f"metadata.json 找不到 timestamp = {timestamp} 的訓練紀錄")
    record = matches[0]
    if record["target"] != "severity":
        raise ValueError(f"目前只支援 severity，此紀錄為 {record['target']}")
    if "test" in record["metrics"] and not force:
        raise ValueError(f"{timestamp} 已有 test 指標，測試集只應評估一次；確定要重新評估請加 --force")
    #--loader 固定讀 DEFAULT_YEARS；年度不同時重建的測試集會不一致
    if record["years"] != list(DEFAULT_YEARS):
        raise ValueError(f"紀錄年度 {record['years']} 與目前 DEFAULT_YEARS {list(DEFAULT_YEARS)} 不同，無法重建相同的測試集")

    #--重建測試集：切分只依年度 × 嚴重度，與欠採樣設定無關
    _, _, test = split_by_year_severity(get_ds_with_adjusted_cols(), record["split_ratios"],
                                        record["random_state"], record["dropped_cols"])
    X_te, y_te = build_xy(test)

    #--載入模型，predict_proba 的第 2 欄為 A1 的預測機率
    model = joblib.load(MODELS_DIR / record["model_file"])
    metrics = severity_metrics(y_te, model.predict_proba(X_te)[:, 1])
    print(f"[severity] test PR-AUC={metrics['average_precision']:.4f}(亂猜≈{metrics['baseline_average_precision']:.4f})，"
          f"recall={metrics['recall']:.3f}，precision={metrics['precision']:.3f}")

    #--寫回訓練紀錄：test 指標、測試集筆數、評估時間
    record["metrics"]["test"] = metrics
    record["rows"]["test"] = len(X_te)
    record["test_evaluated_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_json_record(METADATA_PATH, history)
    print(f"test 指標已寫入 {METADATA_PATH.name}：{timestamp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="版本定案後，在測試集評估一次")
    parser.add_argument("timestamp", help="metadata.json 中訓練紀錄的 timestamp")
    parser.add_argument("--force", action="store_true", help="已有 test 指標時仍重新評估")
    args = parser.parse_args()
    evaluate_test(args.timestamp, force=args.force)
