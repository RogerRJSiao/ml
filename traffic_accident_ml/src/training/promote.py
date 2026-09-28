"""
採用版本登錄(promote.py)
- registry/manifest.json 只記錄各 target 目前採用哪一個版本，入 git，何時改採哪一版可由 git 歷史追溯
- models/metadata.json 保留所有訓練紀錄(實驗日誌)，不入 git
- 採用前須已用 evaluate.py 在 test 評估過，且模型檔存在
- manifest 格式：{"severity": {"timestamp": "20260928T081841Z", "model_file": "severity_20260928T081841Z.joblib"}}

執行：python -m src.training.promote <訓練紀錄 timestamp>
     例如 python -m src.training.promote 20260928T081841Z
"""

import argparse

from src.data_management.common import BASE_DIR, load_json_record, save_json_record
from src.training.train import METADATA_PATH, MODELS_DIR

#--各 target 目前採用版本的登錄檔
MANIFEST_PATH = BASE_DIR / "registry" / "manifest.json"


def promote(timestamp):
    """把指定的訓練紀錄登錄為該 target 的採用版本，寫入 registry/manifest.json"""
    #--找出指定的訓練紀錄
    history = load_json_record(METADATA_PATH) or []
    matches = [r for r in history if r["timestamp"] == timestamp]
    if not matches:
        raise ValueError(f"metadata.json 找不到 timestamp = {timestamp} 的訓練紀錄")
    record = matches[0]
    #--test 是定案前的最後確認，未評估過不可採用
    if "test" not in record["metrics"]:
        raise ValueError(f"{timestamp} 尚未在 test 評估，請先執行 python -m src.training.evaluate {timestamp}")
    #--models/ 不入 git，確認本機有模型檔，避免 manifest 指向不存在的檔案
    if not (MODELS_DIR / record["model_file"]).exists():
        raise FileNotFoundError(f"找不到模型檔 {record['model_file']}")

    #--只覆蓋該 target 的採用版本，其他 target 維持不變
    manifest = load_json_record(MANIFEST_PATH)
    target = record["target"]
    previous = manifest.get(target, {}).get("timestamp")
    manifest[target] = {"timestamp": timestamp, "model_file": record["model_file"]}
    save_json_record(MANIFEST_PATH, manifest)
    print(f"[{target}] 採用版本：{previous or '(無)'} → {timestamp}，已寫入 {MANIFEST_PATH.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="登錄目前採用的模型版本")
    parser.add_argument("timestamp", help="metadata.json 中訓練紀錄的 timestamp")
    args = parser.parse_args()
    promote(args.timestamp)
