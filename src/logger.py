import logging
import os
from datetime import datetime
import multiprocessing

def get_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger that writes to both the console and a timestamped log file.

    Log format:
        2024-01-15 10:23:45 | INFO     | src.train | Epoch 1/30 ...

    Args:
        name (str):    Logger name
        log_dir (str): Directory to write log files into. Created if missing.
        level (int):   Minimum log level. Default: logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if multiprocessing.current_process().name != "MainProcess":
        return logger

    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ------------------------------------------------------------------
    # Shared formatter
    # Same format for both handlers so console and file are consistent.
    # ------------------------------------------------------------------
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ------------------------------------------------------------------
    # Console handler — writes to stdout
    # ------------------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)

    # ------------------------------------------------------------------
    # File handler — writes to logs/<name>_<timestamp>.log
    # ------------------------------------------------------------------
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{name.replace('.', '_')}_{timestamp}.log")

    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    logger.info("Logger initialized — writing to %s", log_filename)
    return logger
