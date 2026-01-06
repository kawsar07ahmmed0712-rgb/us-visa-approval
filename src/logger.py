import logging
import os
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    log_dir = os.path.join(os.getcwd(), "artifacts", "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir, f"log_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs via root logger propagation
    logger.propagate = False

    # Add handlers only once
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s - %(message)s")

        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger
