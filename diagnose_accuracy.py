import json
import os
import pandas as pd


def norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def find_target_column(df: pd.DataFrame) -> str | None:
    # Try common target names (add more if needed)
    candidates = ["case_status", "casestatus", "status"]
    for c in df.columns:
        if norm(c) in candidates:
            return c
    return None


def baseline_accuracy_from_counts(series: pd.Series) -> float:
    vc = series.value_counts(dropna=False)
    if vc.empty:
        return 0.0
    return float(vc.max() / vc.sum())


def main() -> None:
    train_path = os.path.join("artifacts", "data", "train.csv")
    test_path = os.path.join("artifacts", "data", "test.csv")
    eval_path = os.path.join("artifacts", "model_evaluation", "evaluation_report.json")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Run main.py first to generate artifacts/data/train.csv and test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    tcol_train = find_target_column(train_df)
    tcol_test = find_target_column(test_df)

    if tcol_train is None or tcol_test is None:
        print("Could not auto-detect target column.")
        print("Train columns:", train_df.columns.tolist())
        return

    print("Detected target column:", tcol_train)

    # Class distribution
    train_counts = train_df[tcol_train].value_counts(dropna=False)
    test_counts = test_df[tcol_test].value_counts(dropna=False)

    print("\nTrain class counts:")
    print(train_counts)

    print("\nTest class counts:")
    print(test_counts)

    # Baseline (majority class) accuracy
    train_base = baseline_accuracy_from_counts(train_df[tcol_train])
    test_base = baseline_accuracy_from_counts(test_df[tcol_test])

    print(f"\nBaseline accuracy (majority class) -> Train: {train_base:.4f} | Test: {test_base:.4f}")

    # Your model evaluation
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
        metrics = rep.get("test_metrics_thresholded") or rep.get("test_metrics") or {}
        print("\nModel test metrics (from evaluation_report.json):")
        print(json.dumps(metrics, indent=2))
    else:
        print("\nNo evaluation report found yet:", eval_path)

    print("\nInterpretation tip:")
    print("- If baseline test accuracy is already high (e.g., 0.90), 90% is easy by predicting majority.")
    print("- If baseline is ~0.65 and best honest model is ~0.71, then 90% is NOT realistic without leakage or new info/features.")


if __name__ == "__main__":
    main()
