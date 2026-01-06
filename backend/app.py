import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent  # project root (contains src/)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


logger = logging.getLogger("visa.backend")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s visa.backend - %(message)s"))
logger.handlers = [_handler]


# ---------------------------
# Paths
# ---------------------------
APP_DIR = Path(__file__).resolve().parent           # .../backend
ROOT_DIR = APP_DIR.parent                           # .../us-visa-approval

ARTIFACTS_DIR = ROOT_DIR / "artifacts"

PREPROCESSOR_PATH = ARTIFACTS_DIR / "transformation" / "preprocessor.pkl"
MODEL_PATH = ARTIFACTS_DIR / "model_trainer" / "model.pkl"
EVAL_REPORT_PATH = ARTIFACTS_DIR / "model_evaluation" / "evaluation_report.json"
METRICS_PATH = ARTIFACTS_DIR / "model_trainer" / "metrics.json"
TRAIN_CSV_PATH = ARTIFACTS_DIR / "data" / "train.csv"


# ---------------------------
# App
# ---------------------------
app = Flask(__name__)
CORS(app)


# ---------------------------
# Cache (load once)
# ---------------------------
_BUNDLE: Dict[str, Any] = {
    "loaded": False,
    "preprocessor": None,
    "model": None,
    "threshold": 0.5,
    "label_pos": "CERTIFIED",
    "label_neg": "DENIED",
}


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _resolve_threshold() -> float:
    # Prefer evaluation report threshold if present
    rep = _read_json(EVAL_REPORT_PATH)
    if rep and isinstance(rep, dict):
        thr = rep.get("threshold", None)
        if isinstance(thr, (int, float)) and 0 < float(thr) < 1:
            return float(thr)

    met = _read_json(METRICS_PATH)
    if met and isinstance(met, dict):
        thr = met.get("threshold", None)
        if isinstance(thr, (int, float)) and 0 < float(thr) < 1:
            return float(thr)

    return 0.5


def _ensure_loaded() -> Tuple[bool, Optional[str]]:
    """
    Loads preprocessor + model once. Never throws to caller.
    Returns (ok, error_message_if_any).
    """
    if _BUNDLE.get("loaded", False):
        return True, None

    try:
        if not PREPROCESSOR_PATH.exists():
            return False, f"Missing preprocessor.pkl at {PREPROCESSOR_PATH}"
        if not MODEL_PATH.exists():
            return False, f"Missing model.pkl at {MODEL_PATH}"

        logger.info(f"Loading preprocessor: {PREPROCESSOR_PATH}")
        pre = joblib.load(PREPROCESSOR_PATH)

        logger.info(f"Loading model: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)

        thr = _resolve_threshold()

        _BUNDLE["preprocessor"] = pre
        _BUNDLE["model"] = model
        _BUNDLE["threshold"] = float(thr)
        _BUNDLE["loaded"] = True

        return True, None

    except Exception as e:
        logger.exception("Artifact load failed")
        return False, str(e)


def _build_single_row_df(payload: Dict[str, Any], preprocessor: Any) -> pd.DataFrame:
    """
    Builds a one-row DataFrame and aligns columns to what the preprocessor expects (if available).
    Also computes company_age if yr_of_estab exists.
    """
    row = dict(payload)

    # Feature engineering: company_age from yr_of_estab (safe)
    if "yr_of_estab" in row:
        try:
            yr = int(row["yr_of_estab"])
            current_year = pd.Timestamp.now().year
            row["company_age"] = max(0, current_year - yr)
        except Exception:
            # leave as-is
            pass

    df = pd.DataFrame([row])

    # Align to preprocessor expected columns when fitted with DataFrame
    expected = getattr(preprocessor, "feature_names_in_", None)
    if expected is not None:
        expected_cols = list(expected)
        for c in expected_cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[expected_cols]

    return df


def _predict_proba_pos(model: Any, X: Any) -> float:
    """
    Returns probability for the positive class.
    Tries to detect positive class index; falls back to class=1 when possible.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = getattr(model, "classes_", None)

        # Try to locate class "1" if present
        if classes is not None:
            try:
                classes_list = list(classes)
                if 1 in classes_list:
                    idx = classes_list.index(1)
                    return float(proba[idx])

                # Try CERTIFIED string match
                upper_map = [str(c).upper() for c in classes_list]
                if "CERTIFIED" in upper_map:
                    idx = upper_map.index("CERTIFIED")
                    return float(proba[idx])
            except Exception:
                pass

        # fallback: take last column as "positive"
        return float(proba[-1])

    # If no predict_proba, fallback to decision function -> sigmoid-ish
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
        return float(1.0 / (1.0 + np.exp(-score)))

    # ultimate fallback: hard prediction
    pred = int(model.predict(X)[0])
    return 1.0 if pred == 1 else 0.0


@app.get("/health")
def health():
    ok, err = _ensure_loaded()
    # Always return 200 with ok flag (no 500)
    return jsonify(
        {
            "ok": bool(ok),
            "error": err,
            "threshold": _BUNDLE.get("threshold", 0.5),
            "label_pos": _BUNDLE.get("label_pos", "CERTIFIED"),
            "label_neg": _BUNDLE.get("label_neg", "DENIED"),
            "paths": {
                "preprocessor": str(PREPROCESSOR_PATH),
                "model": str(MODEL_PATH),
            },
        }
    )


@app.get("/meta")
def meta():
    """
    Loads dropdown options + numeric ranges from artifacts/data/train.csv if present.
    Always returns 200 with ok flag.
    """
    if not TRAIN_CSV_PATH.exists():
        return jsonify({"ok": False, "error": f"Missing train.csv at {TRAIN_CSV_PATH}"}), 200

    try:
        df = pd.read_csv(TRAIN_CSV_PATH)

        cat_cols = [
            "continent",
            "region_of_employment",
            "unit_of_wage",
            "education_of_employee",
            "has_job_experience",
            "requires_job_training",
            "full_time_position",
        ]
        num_cols = ["no_of_employees", "prevailing_wage", "yr_of_estab"]

        categorical_options: Dict[str, Any] = {}
        for c in cat_cols:
            if c in df.columns:
                vals = (
                    df[c]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                vals = sorted(vals)
                categorical_options[c] = vals

        numeric_ranges: Dict[str, Any] = {}
        for c in num_cols:
            if c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(s) > 0:
                    numeric_ranges[c] = {
                        "min": float(s.min()),
                        "max": float(s.max()),
                        "p50": float(s.median()),
                    }

        return jsonify(
            {
                "ok": True,
                "rows": int(df.shape[0]),
                "source": str(TRAIN_CSV_PATH),
                "categorical_options": categorical_options,
                "numeric_ranges": numeric_ranges,
            }
        ), 200

    except Exception as e:
        logger.exception("Meta failed")
        return jsonify({"ok": False, "error": str(e)}), 200


@app.post("/predict")
def predict():
    ok, err = _ensure_loaded()
    if not ok:
        return jsonify({"ok": False, "error": err}), 200

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "Invalid JSON payload"}), 200

    pre = _BUNDLE["preprocessor"]
    model = _BUNDLE["model"]
    thr = float(_BUNDLE.get("threshold", 0.5))

    try:
        df = _build_single_row_df(payload, pre)
        X = pre.transform(df)
        score_pos = _predict_proba_pos(model, X)

        pred = 1 if score_pos >= thr else 0
        label = _BUNDLE["label_pos"] if pred == 1 else _BUNDLE["label_neg"]

        return jsonify(
            {
                "ok": True,
                "label": label,
                "pred": int(pred),
                "score_pos_class": float(score_pos),
                "threshold": float(thr),
            }
        ), 200

    except Exception as e:
        logger.exception("Predict failed")
        return jsonify({"ok": False, "error": str(e)}), 200


@app.get("/")
def root():
    return jsonify({"ok": True, "service": "visa-backend"}), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
