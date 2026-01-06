import os
from typing import Dict, Tuple, Any

import dill
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except Exception:
    HistGradientBoostingClassifier = None  # type: ignore

from src.config import AppConfig
from src.entity.model_trainer_artifact import ModelTrainerArtifact
from src.entity.transformation_artifact import TransformationArtifact
from src.logger import get_logger
from src.utils import write_json

logger = get_logger("visa.trainer")


def load_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    return data["X"], data["y"]


def predict_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return s
    return model.predict(X).astype(float)


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (y_score >= thr).astype(int)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_pred.mean()),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = float("nan")
    return out


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    IMPORTANT: Don't optimize plain accuracy on imbalanced data.
    We optimize balanced_accuracy (treats both classes fairly).
    """
    best_thr = 0.5
    best_bal_acc = -1.0
    best_metrics: Dict[str, float] = {}

    for thr in np.linspace(0.05, 0.95, 19):
        m = metrics_at_threshold(y_true, y_score, float(thr))
        if m["balanced_accuracy"] > best_bal_acc:
            best_bal_acc = m["balanced_accuracy"]
            best_thr = float(thr)
            best_metrics = m

    return best_thr, best_metrics


def try_smoteenn(X: np.ndarray, y: np.ndarray, rs: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    try:
        from imblearn.combine import SMOTEENN  # type: ignore

        smt = SMOTEENN(random_state=rs, sampling_strategy="minority")
        X2, y2 = smt.fit_resample(X, y)
        return X2, y2, True
    except Exception:
        return X, y, False


class ModelTrainer:
    def __init__(self, config: AppConfig, transformation_artifact: TransformationArtifact) -> None:
        self.config = config
        self.transformation_artifact = transformation_artifact

    def _build_models(self) -> Dict[str, Any]:
        rs = self.config.training.random_state
        models: Dict[str, Any] = {
            "logreg_balanced": LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                n_jobs=None,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=800,
                random_state=rs,
                n_jobs=-1,
                max_features="sqrt",
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=1200,
                random_state=rs,
                n_jobs=-1,
                max_features="sqrt",
            ),
            "knn_9_distance": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("knn", KNeighborsClassifier(n_neighbors=9, weights="distance", n_jobs=-1)),
                ]
            ),
        }

        if HistGradientBoostingClassifier is not None:
            models["hgb"] = HistGradientBoostingClassifier(
                random_state=rs,
                max_depth=4,
                learning_rate=0.06,
                max_iter=600,
            )

        return models

    def run(self) -> ModelTrainerArtifact:
        rs = self.config.training.random_state

        X_train, y_train = load_npz(self.transformation_artifact.train_npz_path)
        X_test, y_test = load_npz(self.transformation_artifact.test_npz_path)

        logger.info(f"Train arrays: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Test arrays:  X={X_test.shape}, y={y_test.shape}")

        # Validation split from training (no leakage)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=rs,
            stratify=y_train if len(np.unique(y_train)) > 1 else None,
        )
        logger.info(f"Internal split: train={X_tr.shape}, val={X_val.shape}")

        models = self._build_models()

        best_name = None
        best_model = None
        best_thr = 0.5
        best_val_balacc = -1.0

        report_all: Dict[str, Any] = {}

        for name, model in models.items():
            # Resample ONLY train fold
            X_fit, y_fit, used = try_smoteenn(X_tr, y_tr, rs)
            if used:
                logger.info(f"{name}: SMOTEENN applied on train-fold -> X={X_fit.shape}, y={y_fit.shape}")

            logger.info(f"Training model: {name}")
            model.fit(X_fit, y_fit)

            val_scores = predict_scores(model, X_val)
            thr, val_metrics = find_best_threshold(y_val, val_scores)

            report_all[name] = {
                "best_threshold": thr,
                "val_metrics_at_best_threshold": val_metrics,
                "smoteenn_train_fold": used,
            }
            logger.info(f"{name} best_thr={thr} | val_metrics={val_metrics}")

            if val_metrics["balanced_accuracy"] > best_val_balacc:
                best_val_balacc = val_metrics["balanced_accuracy"]
                best_name = name
                best_model = model
                best_thr = thr

        if best_model is None or best_name is None:
            raise RuntimeError("No model trained successfully.")

        # IMPORTANT: Refit best model on FULL training with same resampling style
        X_full_fit, y_full_fit, used_full = try_smoteenn(X_train, y_train, rs)
        logger.info(f"Refitting best model on full train: {best_name} | smoteenn={used_full}")
        best_model.fit(X_full_fit, y_full_fit)

        # Final test metrics with chosen threshold
        test_scores = predict_scores(best_model, X_test)
        test_metrics = metrics_at_threshold(y_test, test_scores, best_thr)

        trainer_dir = os.path.join(os.getcwd(), self.config.artifacts.root_dir, "model_trainer")
        os.makedirs(trainer_dir, exist_ok=True)

        model_path = os.path.join(trainer_dir, "model.pkl")
        metrics_path = os.path.join(trainer_dir, "metrics.json")

        with open(model_path, "wb") as f:
            dill.dump(best_model, f)

        out = {
            "best_model_name": best_name,
            "best_threshold": best_thr,
            "best_val_balanced_accuracy": best_val_balacc,
            "all_models": report_all,
            "test_metrics_with_best_threshold": test_metrics,
            "note": "We optimize balanced_accuracy to avoid 'predict-all-one' behavior. Accuracy alone is misleading on imbalanced data.",
        }
        write_json(metrics_path, out)

        logger.info(f"Best model: {best_name} | best_val_balacc={best_val_balacc} | best_thr={best_thr}")
        logger.info(f"Test metrics (thresholded): {test_metrics}")
        logger.info(f"Saved model: {model_path}")
        logger.info(f"Saved metrics: {metrics_path}")

        return ModelTrainerArtifact(
            model_path=model_path,
            metrics_path=metrics_path,
            best_model_name=best_name,
        )
