import os
import json
from datetime import datetime

import dill

from src.config import build_config
from src.utils import read_yaml


def load_threshold(metrics_path: str) -> float:
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get("best_threshold", 0.5))


def main() -> None:
    # Silence loky warning for this script too
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

    preprocessor_path = os.path.join("artifacts", "transformation", "preprocessor.pkl")
    model_path = os.path.join("artifacts", "model_trainer", "model.pkl")
    metrics_path = os.path.join("artifacts", "model_trainer", "metrics.json")

    if not os.path.exists(preprocessor_path) or not os.path.exists(model_path) or not os.path.exists(metrics_path):
        raise FileNotFoundError("Run main.py first to generate preprocessor/model/metrics artifacts.")

    with open(preprocessor_path, "rb") as f:
        preprocessor = dill.load(f)

    with open(model_path, "rb") as f:
        model = dill.load(f)

    threshold = load_threshold(metrics_path)

    cfg_dict = read_yaml("config/config.yaml")
    cfg = build_config(cfg_dict)

    bundle = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": cfg_dict.get("training", {}).get("model_name", "best_model"),
        "threshold": float(threshold),
        "pos_label": cfg.training.positive_class,
        "neg_label": cfg.training.negative_class,
        "target_column": cfg.training.target_column,
        "preprocessor": preprocessor,
        "model": model,
    }

    out_dir = os.path.join("artifacts", "model_bundle")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_bundle.pkl")

    with open(out_path, "wb") as f:
        dill.dump(bundle, f)

    print(f"Saved bundle: {out_path}")


if __name__ == "__main__":
    main()
