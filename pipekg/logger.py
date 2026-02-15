import logging
from pathlib import Path
from typing import Optional


_DEF_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(logging.Formatter(_DEF_FMT))
        logger.addHandler(sh)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(log_file) for h in logger.handlers):
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(_DEF_FMT))
            logger.addHandler(fh)

    return logger
