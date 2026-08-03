"""Il logger dell'installer.

Due canali: uno a schermo, breve, per chi guarda l'installazione andare
avanti; uno su file, completo, per chi deve capire dopo perche' un passo e'
fallito su una macchina che non e' la propria. --dry-run non chiama mai
questo modulo, per non lasciare traccia sul disco.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

LOGGER_NAME = "dfl_install"


def setup_logging(log_file: Path) -> logging.Logger:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()  # rilanciare setup_logging non deve accumulare handler

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)

    return logger
