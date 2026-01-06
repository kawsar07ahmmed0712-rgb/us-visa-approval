import json
import os
import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except Exception:
    HistGradientBoostingClassifier = None  # type: ignore


def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    return d["X"], d["y"]


def predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return s
    return model.predict(X).astype(float)


def pick_threshold_bal_acc(y_true, y_score):
    best_thr = 0.5
    best_bal = -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        y_pred = (y_score >= thr).astype(int)
        bal = balanced_accuracy_score(y_true, y_pred)
        if bal > best_bal:
            best_bal = bal
            best_thr = float(thr)
    return best_thr


def build_model(best_name: str, random_state: int = 42):
    # match your trainer naming
    if best_name == "hgb":
        if HistGradientBoostingClassifier is None:
            raise RuntimeError("HistGradientBoostingClassifier not available in your sklearn.")
        return HistGradientBoostingClassifier(
            random_state=random_state,
            max_depth=4,
            learning_rate=0.06,
            max_iter=600,
        )

    if best_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=800,
            random_state=random_state,
            n_jobs=-1,
            max_features="sqrt",
        )

    if best_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=1200,
            random_state=random_state,
            n_jobs=-1,
            max_features="sqrt",
        )

    if best_name == "knn_9_distance":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=9, weights="distance", n_jobs=-1)),
            ]
        )

    # fallback: hgb if possible
    if HistGradientBoostingClassifier is not None:
        return HistGradientBoostingClassifier(random_state=random_state)
    raise RuntimeError(f"Unknown model name: {best_name}")


def evaluate(model, X_test, y_test, thr):
    score = predict_scores(model, X_test)
    pred = (score >= thr).astype(int)

    out = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_test, score))
    except Exception:
        out["roc_auc"] = None
    return out


def main():
    train_npz = os.path.join("artifacts", "transformation", "train.npz")
    test_npz = os.path.join("artifacts", "transformation", "test.npz")
    metrics_json = os.path.join("artifacts", "model_trainer", "metrics.json")

    if not (os.path.exists(train_npz) and os.path.exists(test_npz) and os.path.exists(metrics_json)):
        raise FileNotFoundError("Run main.py first to generate artifacts (train.npz/test.npz/metrics.json).")

    X_train, y_train = load_npz(train_npz)
    X_test, y_test = load_npz(test_npz)

    with open(metrics_json, "r", encoding="utf-8") as f:
        m = json.load(f)

    best_name = m.get("best_model_name", "hgb")
    rs = 42

    # Baseline (majority class)
    p = float(np.mean(y_test))
    baseline_acc = max(p, 1 - p)
    print(f"Test positive-rate={p:.3f} -> baseline accuracy (predict-majority)={baseline_acc:.3f}")
    print(f"Using model family: {best_name}\n")

    n = len(y_train)
    sizes = [1000, 3000, 6000, 12000, n]
    sizes = [s for s in sizes if s < n] + [n]

    print("train_size | thr(val) | test_acc | test_bal_acc | test_auc")
    print("-" * 60)

    for s in sizes:
        # stratified subset
        idx_pos = np.where(y_train == 1)[0]
        idx_neg = np.where(y_train == 0)[0]

        # sample proportionally
        s_pos = int(round(s * (len(idx_pos) / n)))
        s_neg = s - s_pos

        rng = np.random.default_rng(rs + s)
        sub_pos = rng.choice(idx_pos, size=min(s_pos, len(idx_pos)), replace=False)
        sub_neg = rng.choice(idx_neg, size=min(s_neg, len(idx_neg)), replace=False)
        sub_idx = np.concatenate([sub_pos, sub_neg])
        rng.shuffle(sub_idx)

        X_sub = X_train[sub_idx]
        y_sub = y_train[sub_idx]

        # internal split for threshold tuning
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=rs, stratify=y_sub if len(np.unique(y_sub)) > 1 else None
        )

        model = build_model(best_name, random_state=rs)
        model.fit(X_tr, y_tr)

        val_score = predict_scores(model, X_val)
        thr = pick_threshold_bal_acc(y_val, val_score)

        # refit on full subset
        model.fit(X_sub, y_sub)

        out = evaluate(model, X_test, y_test, thr)
        print(
            f"{s:9d} | {thr:7.2f} | {out['accuracy']:.3f}   | {out['balanced_accuracy']:.3f}        | {out['roc_auc'] if out['roc_auc'] is not None else 'NA'}"
        )

    print("\nHow to read:")
    print("- If test_acc keeps rising with train_size -> more data will help.")
    print("- If test_acc plateaus ~0.70-0.75 -> data/feature limit; you need better features or different dataset.")


if __name__ == "__main__":
    main()
