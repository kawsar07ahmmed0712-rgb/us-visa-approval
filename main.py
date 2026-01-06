from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.config import build_config
from src.logger import get_logger
from src.utils import read_yaml

logger = get_logger("visa")


def main() -> None:
    cfg_dict = read_yaml("config/config.yaml")
    cfg = build_config(cfg_dict)

    logger.info("Starting pipeline: Data Ingestion")
    ingestion_artifact = DataIngestion(cfg).run()

    logger.info("Starting pipeline: Data Validation")
    validation_artifact = DataValidation(cfg, ingestion_artifact).run()

    logger.info(f"Validation done: {validation_artifact}")


if __name__ == "__main__":
    main()
