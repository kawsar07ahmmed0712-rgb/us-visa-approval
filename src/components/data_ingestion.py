import os
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import AppConfig
from src.entity.ingestion_artifact import IngestionArtifact
from src.logger import get_logger

logger = get_logger("visa.ingestion")


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


class DataIngestion:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run(self) -> IngestionArtifact:
        csv_path = self.config.dataset.csv_path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at: {csv_path}")

        logger.info(f"Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded data shape: {df.shape}")

        data_dir = os.path.join(os.getcwd(), self.config.artifacts.root_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        raw_path = os.path.join(data_dir, "raw.csv")
        train_path = os.path.join(data_dir, "train.csv")
        test_path = os.path.join(data_dir, "test.csv")

        df.to_csv(raw_path, index=False)
        logger.info(f"Saved raw data to: {raw_path}")

        configured_target = self.config.training.target_column
        resolved_target = resolve_column_name(df, configured_target)

        stratify_col = None
        if resolved_target is not None:
            y = df[resolved_target]
            if y.notna().any() and y.nunique(dropna=True) >= 2:
                stratify_col = y
            else:
                logger.warning("Target exists but stratify not possible (not enough classes / all null).")
        else:
            logger.warning(
                f"Target column '{configured_target}' not found. Available columns: {df.columns.tolist()}"
            )

        train_df, test_df = train_test_split(
            df,
            test_size=self.config.training.test_size,
            random_state=self.config.training.random_state,
            stratify=stratify_col,
        )

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Saved train data to: {train_path} | shape={train_df.shape}")
        logger.info(f"Saved test data to: {test_path} | shape={test_df.shape}")

        return IngestionArtifact(
            raw_data_path=raw_path,
            train_data_path=train_path,
            test_data_path=test_path,
        )
