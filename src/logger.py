import logging
import os

os.makedirs("logs", exist_ok=True)

def get_logger(file_name, log_file):
    logger = logging.getLogger(file_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(f"logs/{log_file}")
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        fh.setFormatter(formatter)

        logger.addHandler(fh)

    return logger 