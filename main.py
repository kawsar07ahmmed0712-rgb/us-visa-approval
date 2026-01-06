from src.logger import get_logger
from src.utils import read_yaml
from src.config import build_config

logger = get_logger("visa")

def main() -> None:
    cfg_dict = read_yaml("config/config.yaml")
    cfg = build_config(cfg_dict)

    logger.info("Config loaded successfully.")
    logger.info(f"CSV Path: {cfg.dataset.csv_path}")
    logger.info(f"Target: {cfg.training.target_column}")
    logger.info(f"Classes: {cfg.training.positive_class} vs {cfg.training.negative_class}")

if __name__ == "__main__":
    main()
