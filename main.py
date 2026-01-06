from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

from src.config import build_config
from src.logger import get_logger
from src.utils import read_yaml

import os

# Silence joblib/loky physical core detection warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

logger = get_logger("visa")

def main() -> None:
    cfg_dict = read_yaml("config/config.yaml")
    cfg = build_config(cfg_dict)

    logger.info("Starting pipeline: Data Ingestion")
    ingestion_artifact = DataIngestion(cfg).run()

    logger.info("Starting pipeline: Data Validation")
    _ = DataValidation(cfg, ingestion_artifact).run()

    logger.info("Starting pipeline: Data Transformation")
    transformation_artifact = DataTransformation(cfg, ingestion_artifact).run()

    logger.info("Starting pipeline: Model Trainer")
    trainer_artifact = ModelTrainer(cfg, transformation_artifact).run()

    # HARD GUARD (so evaluation never receives None)
    if trainer_artifact is None:
        raise RuntimeError("ModelTrainer.run() returned None. Check model_trainer.py return statement.")

    logger.info(f"Trainer artifact: {trainer_artifact}")

    logger.info("Starting pipeline: Model Evaluation")
    eval_artifact = ModelEvaluation(cfg, transformation_artifact, trainer_artifact).run()

    logger.info(f"Pipeline done. Evaluation: {eval_artifact}")

if __name__ == "__main__":
    main()
