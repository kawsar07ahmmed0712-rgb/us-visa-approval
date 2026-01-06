import argparse
import json
import os
from typing import Any, Dict, Optional

import dill
import numpy as np
import pandas as pd

from src.config import build_config
from src.utils import read_yaml
from src.components.data_transformation import apply_basic_feature_engineering, resolve_column_name
import os
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)


def load_threshold(metrics_path: str) -> float:
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get("best_threshold", 0.5))


def predict_scores(model: Any, X: np.ndarray) -> np.ndarray:
    """Return probability/score for positive class."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return s
    return model.predict(X).astype(float)

def load_artifacts() -> Dict[str, Any]:
    bundle_path = os.path.join("artifacts", "model_bundle", "model_bundle.pkl")

    # 1) Prefer bundle (showcase mode)
    if os.path.exists(bundle_path):
        with open(bundle_path, "rb") as f:
            bundle = dill.load(f)
        return {
            "preprocessor": bundle["preprocessor"],
            "model": bundle["model"],
            "threshold": float(bundle["threshold"]),
            "pos_label": bundle["pos_label"],
            "neg_label": bundle["neg_label"],
            "target_column": bundle["target_column"],
        }

    # 2) Fallback to individual artifacts
    preprocessor_path = os.path.join("artifacts", "transformation", "preprocessor.pkl")
    model_path = os.path.join("artifacts", "model_trainer", "model.pkl")
    metrics_path = os.path.join("artifacts", "model_trainer", "metrics.json")

    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Missing: {preprocessor_path} (run main.py first)")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing: {model_path} (run main.py first)")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Missing: {metrics_path} (run main.py first)")

    with open(preprocessor_path, "rb") as f:
        preprocessor = dill.load(f)

    with open(model_path, "rb") as f:
        model = dill.load(f)

    threshold = load_threshold(metrics_path)

    cfg_dict = read_yaml("config/config.yaml")
    cfg = build_config(cfg_dict)

    return {
        "preprocessor": preprocessor,
        "model": model,
        "threshold": threshold,
        "pos_label": cfg.training.positive_class,
        "neg_label": cfg.training.negative_class,
        "target_column": cfg.training.target_column,
    }

def prepare_input_df(df: pd.DataFrame, target_column_cfg: str) -> pd.DataFrame:
    """Make input consistent with training pipeline."""
    df = df.copy()

    # Apply same basic FE used before fitting the preprocessor
    df = apply_basic_feature_engineering(df)

    # Drop target column if present (case-insensitive resolve)
    tcol = resolve_column_name(df, target_column_cfg)
    if tcol is not None and tcol in df.columns:
        df = df.drop(columns=[tcol])

    return df


def predict_one_row(df_row: pd.DataFrame) -> Dict[str, Any]:
    art = load_artifacts()

    df_row = prepare_input_df(df_row, art["target_column"])

    X = art["preprocessor"].transform(df_row)
    score = float(predict_scores(art["model"], X)[0])

    pred = 1 if score >= art["threshold"] else 0
    label = art["pos_label"] if pred == 1 else art["neg_label"]

    return {
        "label": label,
        "pred": pred,
        "score_pos_class": score,
        "threshold": float(art["threshold"]),
    }


def read_row_from_csv(path: str, row_index: int) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if row_index < 0 or row_index >= len(df):
        raise ValueError(f"row-index out of range. Given={row_index}, rows={len(df)}")
    return df.iloc[[row_index]].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-test", type=int, default=None, help="Pick a row from artifacts/data/test.csv")
    parser.add_argument("--input-csv", type=str, default=None, help="Your CSV path for prediction")
    parser.add_argument("--row-index", type=int, default=0, help="Row index for --input-csv")
    args = parser.parse_args()

    if args.from_test is None and args.input_csv is None:
        raise SystemExit(
            "Use one:\n"
            "  python predict.py --from-test 0\n"
            "  python predict.py --input-csv \"path/to/file.csv\" --row-index 0"
        )

    if args.from_test is not None:
        test_csv = os.path.join("artifacts", "data", "test.csv")
        row_df = read_row_from_csv(test_csv, args.from_test)
    else:
        row_df = read_row_from_csv(args.input_csv, args.row_index)

    result = predict_one_row(row_df)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
