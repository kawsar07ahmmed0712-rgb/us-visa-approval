import os
from typing import Optional

import pandas as pd

from src.config import AppConfig
from src.entity.ingestion_artifact import IngestionArtifact
from src.entity.validation_artifact import ValidationArtifact
from src.logger import get_logger
from src.utils import write_json, write_yaml

logger = get_logger("visa.validation")


def resolve_column_name(df: pd.DataFrame, configured_name: str) -> Optional[str]:
    """Resolve a column name even if case/spacing/underscore differs."""
    if configured_name in df.columns:
        return configured_name

    def norm(s: str) -> str:
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    target_norm = norm(configured_name)
    for c in df.columns:
        if norm(c) == target_norm:
            return c
    return None


class DataValidation:
    def __init__(self, config: AppConfig, ingestion_artifact: IngestionArtifact) -> None:
        self.config = config
        self.ingestion_artifact = ingestion_artifact

    def _infer_schema(self, df: pd.DataFrame, target_col: str) -> dict:
        cols = list(df.columns)
        dtypes = {c: str(df[c].dtype) for c in cols}

        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in cols if c not in numeric_cols]

        return {
            "columns": cols,
            "dtypes": dtypes,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "target_column": target_col,
        }

    def run(self) -> ValidationArtifact:
        train_path = self.ingestion_artifact.train_data_path
        test_path = self.ingestion_artifact.test_data_path

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError("Train/Test data not found. Run ingestion first.")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        configured_target = self.config.training.target_column
        target_train = resolve_column_name(train_df, configured_target)
        target_test = resolve_column_name(test_df, configured_target)

        if target_train is None or target_test is None:
            raise ValueError(
                f"Target column not found. Configured='{configured_target}'. "
                f"Train columns={train_df.columns.tolist()}"
            )

        # Use resolved name from train
        target_col = target_train

        validation_dir = os.path.join(os.getcwd(), self.config.artifacts.root_dir, "validation")
        os.makedirs(validation_dir, exist_ok=True)

        # Save inferred schema
        schema = self._infer_schema(train_df, target_col)
        schema_path = os.path.join(validation_dir, "schema_inferred.yaml")
        write_yaml(schema_path, schema)

        # Missing report
        missing_train = (train_df.isna().mean() * 100).round(2)
        missing_test = (test_df.isna().mean() * 100).round(2)

        missing_report_df = pd.DataFrame(
            {
                "missing_train_percent": missing_train,
                "missing_test_percent": missing_test,
            }
        ).sort_values(by="missing_train_percent", ascending=False)

        missing_report_path = os.path.join(validation_dir, "missing_report.csv")
        missing_report_df.to_csv(missing_report_path, index=True)

        # Duplicates
        dup_train = int(train_df.duplicated().sum())
        dup_test = int(test_df.duplicated().sum())

        # Column consistency
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        extra_in_train = sorted(list(train_cols - test_cols))
        extra_in_test = sorted(list(test_cols - train_cols))

        status = True
        issues = []

        if extra_in_train or extra_in_test:
            status = False
            issues.append(
                {
                    "type": "column_mismatch",
                    "extra_in_train": extra_in_train,
                    "extra_in_test": extra_in_test,
                }
            )

        # Target class sanity (warning only)
        pos = self.config.training.positive_class
        neg = self.config.training.negative_class
        train_classes = set(train_df[target_col].dropna().astype(str).unique().tolist())
        if (pos not in train_classes) or (neg not in train_classes):
            logger.warning(
                f"Expected classes not fully found in train target. "
                f"Expected: {pos},{neg} | Found sample: {sorted(list(train_classes))[:10]}"
            )

        report = {
            "status": status,
            "train_shape": [int(train_df.shape[0]), int(train_df.shape[1])],
            "test_shape": [int(test_df.shape[0]), int(test_df.shape[1])],
            "configured_target": configured_target,
            "resolved_target": target_col,
            "target_unique_train_sample": sorted(list(train_classes))[:20],
            "duplicates": {"train": dup_train, "test": dup_test},
            "issues": issues,
        }

        report_path = os.path.join(validation_dir, "validation_report.json")
        write_json(report_path, report)

        logger.info(f"Validation status: {status}")
        logger.info(f"Saved schema: {schema_path}")
        logger.info(f"Saved missing report: {missing_report_path}")
        logger.info(f"Saved validation report: {report_path}")

        if not status:
            raise ValueError("Validation failed. Check artifacts/validation/validation_report.json")

        return ValidationArtifact(
            report_path=report_path,
            missing_report_path=missing_report_path,
            schema_path=schema_path,
            status=status,
        )
