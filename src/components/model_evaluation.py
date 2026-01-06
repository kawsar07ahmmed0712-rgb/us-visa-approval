import os
from typing import Dict, Any, Tuple

import dill
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

from src.config import AppConfig
from src.entity.model_evaluation_artifact import ModelEvaluationArtifact
from src.entity.model_trainer_artifact import ModelTrainerArtifact
from src.entity.transformation_artifact import TransformationArtifact
from src.logger import get_logger
from src.utils import write_json

logger = get_logger("visa.evaluation")


def load_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    return data["X"], data["y"]


def load_threshold(metrics_path: str) -> float:
    # Lightweight JSON read (no new dependency)
    import json
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get("best_threshold", 0.5))


def predict_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return scores
    return model.predict(X).astype(float)


def evaluate_with_threshold(model: Any, X: np.ndarray, y: np.ndarray, thr: float) -> Dict[str, Any]:
    score = predict_scores(model, X)
    pred = (score >= thr).astype(int)

    out: Dict[str, Any] = {
        "threshold": thr,
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, score))
    except Exception:
        out["roc_auc"] = None
    return out


class ModelEvaluation:
    def __init__(
        self,
        config: AppConfig,
        transformation_artifact: TransformationArtifact,
        trainer_artifact: ModelTrainerArtifact,
    ) -> None:
        self.config = config
        self.transformation_artifact = transformation_artifact
        self.trainer_artifact = trainer_artifact

    def run(self) -> ModelEvaluationArtifact:
        X_test, y_test = load_npz(self.transformation_artifact.test_npz_path)

        if not os.path.exists(self.trainer_artifact.model_path):
            raise FileNotFoundError(f"Model not found: {self.trainer_artifact.model_path}")
        if not os.path.exists(self.trainer_artifact.metrics_path):
            raise FileNotFoundError(f"Metrics not found: {self.trainer_artifact.metrics_path}")

        with open(self.trainer_artifact.model_path, "rb") as f:
            model = dill.load(f)

        thr = load_threshold(self.trainer_artifact.metrics_path)
        metrics = evaluate_with_threshold(model, X_test, y_test, thr)

        eval_dir = os.path.join(os.getcwd(), self.config.artifacts.root_dir, "model_evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        report_path = os.path.join(eval_dir, "evaluation_report.json")
        report = {
            "best_model_name": self.trainer_artifact.best_model_name,
            "test_metrics_thresholded": metrics,
        }
        write_json(report_path, report)

        logger.info(f"Saved evaluation report: {report_path}")
        logger.info(f"Test metrics (thresholded): {metrics}")

        return ModelEvaluationArtifact(evaluation_report_path=report_path)
