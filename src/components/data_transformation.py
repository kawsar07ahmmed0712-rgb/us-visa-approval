import os
from datetime import date
from typing import Optional, Tuple, List, Dict, Any

import dill
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer

from src.config import AppConfig
from src.entity.ingestion_artifact import IngestionArtifact
from src.entity.transformation_artifact import TransformationArtifact
from src.logger import get_logger
from src.utils import write_json

logger = get_logger("visa.transformation")


def resolve_column_name(df: pd.DataFrame, configured_name: str) -> Optional[str]:
    if configured_name in df.columns:
        return configured_name

    def norm(s: str) -> str:
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    t = norm(configured_name)
    for c in df.columns:
        if norm(c) == t:
            return c
    return None


def normalize_label(x: object) -> str:
    return str(x).strip().upper()


def filter_and_map_target(
    df: pd.DataFrame, target_col: str, pos_label: str, neg_label: str
) -> Tuple[pd.DataFrame, np.ndarray]:
    pos_n = normalize_label(pos_label)
    neg_n = normalize_label(neg_label)

    y_norm = df[target_col].map(normalize_label)
    keep = y_norm.isin([pos_n, neg_n])

    dropped = int((~keep).sum())
    if dropped > 0:
        logger.warning(f"Dropping {dropped} rows because target not in [{pos_n},{neg_n}].")

    df2 = df.loc[keep].copy()
    y2 = y_norm.loc[keep]
    y = (y2 == pos_n).astype(int).to_numpy()
    return df2, y


def apply_basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safe basic FE (no leakage):
    - Drop case_id if exists
    - company_age from yr_of_estab
    """
    df = df.copy()

    if "case_id" in df.columns:
        df.drop(columns=["case_id"], inplace=True)

    if "yr_of_estab" in df.columns:
        current_year = date.today().year
        df["company_age"] = current_year - pd.to_numeric(df["yr_of_estab"], errors="coerce")
        df.drop(columns=["yr_of_estab"], inplace=True)

    return df


class LeakageSafeFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates stronger numeric features, but learns any group statistics ONLY from training data.

    Features:
    - annual_wage from prevailing_wage + unit_of_wage (approx)
    - log1p transforms for skewed fields
    - wage_per_employee
    - wage_vs_region_median (ratio)
    - wage_vs_continent_median (ratio)
    """

    def __init__(
        self,
        wage_col: str = "prevailing_wage",
        wage_unit_col: str = "unit_of_wage",
        region_col: str = "region_of_employment",
        continent_col: str = "continent",
        employees_col: str = "no_of_employees",
    ) -> None:
        self.wage_col = wage_col
        self.wage_unit_col = wage_unit_col
        self.region_col = region_col
        self.continent_col = continent_col
        self.employees_col = employees_col

        self.region_median_: Dict[str, float] = {}
        self.continent_median_: Dict[str, float] = {}
        self.global_median_: float = 0.0

    def _annualize(self, wage: pd.Series, unit: pd.Series) -> pd.Series:
        # conservative conversions
        # If unit missing/unknown -> treat as already annual
        unit_norm = unit.astype(str).str.strip().str.lower()
        w = pd.to_numeric(wage, errors="coerce")

        factor = pd.Series(np.ones(len(w), dtype=float), index=w.index)
        factor[unit_norm.str.contains("hour")] = 2080.0
        factor[unit_norm.str.contains("week")] = 52.0
        factor[unit_norm.str.contains("month")] = 12.0
        factor[unit_norm.str.contains("year")] = 1.0

        return w * factor

    def fit(self, X: pd.DataFrame, y: Any = None) -> "LeakageSafeFeatureEngineer":
        X2 = X.copy()

        if self.wage_col in X2.columns and self.wage_unit_col in X2.columns:
            annual = self._annualize(X2[self.wage_col], X2[self.wage_unit_col])
        elif self.wage_col in X2.columns:
            annual = pd.to_numeric(X2[self.wage_col], errors="coerce")
        else:
            annual = pd.Series([np.nan] * len(X2), index=X2.index)

        X2["_annual_wage_tmp"] = annual
        self.global_median_ = float(np.nanmedian(X2["_annual_wage_tmp"].to_numpy()))

        # group medians from TRAIN only
        if self.region_col in X2.columns:
            grp = X2.groupby(self.region_col)["_annual_wage_tmp"].median()
            self.region_median_ = {str(k): float(v) for k, v in grp.items() if pd.notna(v)}

        if self.continent_col in X2.columns:
            grp = X2.groupby(self.continent_col)["_annual_wage_tmp"].median()
            self.continent_median_ = {str(k): float(v) for k, v in grp.items() if pd.notna(v)}

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X2 = X.copy()

        # Annual wage
        if self.wage_col in X2.columns and self.wage_unit_col in X2.columns:
            X2["annual_wage"] = self._annualize(X2[self.wage_col], X2[self.wage_unit_col])
        elif self.wage_col in X2.columns:
            X2["annual_wage"] = pd.to_numeric(X2[self.wage_col], errors="coerce")
        else:
            X2["annual_wage"] = np.nan

        # wage_per_employee
        if self.employees_col in X2.columns:
            emp = pd.to_numeric(X2[self.employees_col], errors="coerce")
            X2["wage_per_employee"] = X2["annual_wage"] / (emp.clip(lower=0) + 1.0)
        else:
            X2["wage_per_employee"] = np.nan

        # region/continent ratio features (use TRAIN medians only)
        if self.region_col in X2.columns:
            reg_med = X2[self.region_col].astype(str).map(self.region_median_)
            reg_med = reg_med.fillna(self.global_median_)
            X2["wage_vs_region_median"] = X2["annual_wage"] / (reg_med + 1e-9)
        else:
            X2["wage_vs_region_median"] = np.nan

        if self.continent_col in X2.columns:
            con_med = X2[self.continent_col].astype(str).map(self.continent_median_)
            con_med = con_med.fillna(self.global_median_)
            X2["wage_vs_continent_median"] = X2["annual_wage"] / (con_med + 1e-9)
        else:
            X2["wage_vs_continent_median"] = np.nan

        # log features (skew fix)
        for col in ["annual_wage", "wage_per_employee", "company_age", "no_of_employees", "prevailing_wage"]:
            if col in X2.columns:
                v = pd.to_numeric(X2[col], errors="coerce").clip(lower=0)
                X2[f"log1p_{col}"] = np.log1p(v)

        return X2


class DataTransformation:
    def __init__(self, config: AppConfig, ingestion_artifact: IngestionArtifact) -> None:
        self.config = config
        self.ingestion_artifact = ingestion_artifact

    def _build_preprocessor(self, X: pd.DataFrame) -> Pipeline:
        # Categorical columns to OneHot (better than "random ordinal order")
        oh_columns = [
            "continent",
            "unit_of_wage",
            "region_of_employment",
            "education_of_employee",
            "has_job_experience",
            "requires_job_training",
            "full_time_position",
        ]
        oh_cols = [c for c in oh_columns if c in X.columns]

        # PowerTransform for heavily skewed numeric columns (after FE)
        power_cols_candidates = ["company_age", "no_of_employees", "annual_wage", "wage_per_employee"]
        power_cols = [c for c in power_cols_candidates if c in X.columns]

        # Remaining numeric columns -> StandardScaler
        numeric_cols_all = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        other_num_cols = [c for c in numeric_cols_all if c not in power_cols]

        logger.info(
            f"Columns picked | onehot={oh_cols} | power={power_cols} | other_num={len(other_num_cols)}"
        )

        # OneHotEncoder safe for unseen categories
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        onehot_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", ohe),
            ]
        )

        power_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("power", PowerTransformer(method="yeo-johnson")),
            ]
        )

        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        ct = ColumnTransformer(
            transformers=[
                ("onehot", onehot_pipe, oh_cols),
                ("power", power_pipe, power_cols),
                ("num", num_pipe, other_num_cols),
            ],
            remainder="drop",
        )

        # IMPORTANT: FeatureEngineer is INSIDE this Pipeline (fit only on train)
        pipeline = Pipeline(
            steps=[
                ("fe", LeakageSafeFeatureEngineer()),
                ("ct", ct),
            ]
        )
        return pipeline

    def run(self) -> TransformationArtifact:
        train_df = pd.read_csv(self.ingestion_artifact.train_data_path)
        test_df = pd.read_csv(self.ingestion_artifact.test_data_path)

        train_df = apply_basic_feature_engineering(train_df)
        test_df = apply_basic_feature_engineering(test_df)

        configured_target = self.config.training.target_column
        target_col = resolve_column_name(train_df, configured_target)
        if target_col is None:
            raise ValueError(
                f"Target column not found. Configured='{configured_target}'. "
                f"Train columns={train_df.columns.tolist()}"
            )

        train_df, y_train = filter_and_map_target(
            train_df, target_col, self.config.training.positive_class, self.config.training.negative_class
        )
        test_df, y_test = filter_and_map_target(
            test_df, target_col, self.config.training.positive_class, self.config.training.negative_class
        )

        X_train = train_df.drop(columns=[target_col])
        X_test = test_df.drop(columns=[target_col])

        preprocessor = self._build_preprocessor(X_train)

        logger.info("Fitting preprocessor (with feature engineering) on train data...")
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        trans_dir = os.path.join(os.getcwd(), self.config.artifacts.root_dir, "transformation")
        os.makedirs(trans_dir, exist_ok=True)

        preprocessor_path = os.path.join(trans_dir, "preprocessor.pkl")
        train_npz_path = os.path.join(trans_dir, "train.npz")
        test_npz_path = os.path.join(trans_dir, "test.npz")
        feature_names_path = os.path.join(trans_dir, "feature_names.json")

        with open(preprocessor_path, "wb") as f:
            dill.dump(preprocessor, f)

        np.savez_compressed(train_npz_path, X=X_train_t, y=y_train)
        np.savez_compressed(test_npz_path, X=X_test_t, y=y_test)

        feature_names: List[str] = []
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except Exception:
            logger.warning("Could not extract feature names from preprocessor.")

        write_json(feature_names_path, {"feature_names": feature_names})

        logger.info(f"Saved preprocessor: {preprocessor_path}")
        logger.info(f"Saved train arrays: {train_npz_path} | X={getattr(X_train_t, 'shape', None)} y={y_train.shape}")
        logger.info(f"Saved test arrays: {test_npz_path} | X={getattr(X_test_t, 'shape', None)} y={y_test.shape}")

        return TransformationArtifact(
            preprocessor_path=preprocessor_path,
            train_npz_path=train_npz_path,
            test_npz_path=test_npz_path,
            feature_names_path=feature_names_path,
        )
