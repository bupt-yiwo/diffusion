import logging
import os
import sys
from datetime import datetime


def setup_logger(logdir=None, name="default", level="INFO"):
    """
    Setup a logger that logs to both console and file (if logdir is provided).
    
    Args:
        logdir (str or None): path to save the log file
        name (str): logger name
        level (str): logging level (e.g., "INFO", "DEBUG")

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if logdir:
        os.makedirs(logdir, exist_ok=True)
        log_path = os.path.join(logdir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
