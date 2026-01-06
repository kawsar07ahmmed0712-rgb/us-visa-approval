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

    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s - %(message)s")

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
