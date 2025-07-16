import logging
import os
from datetime import datetime

def init_app_logger(log_name="app"):
  log_dir = "./logs/app"
  os.makedirs(log_dir, exist_ok=True)
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  log_path = os.path.join(log_dir, f"{log_name}_{ts}.log")

  logger = logging.getLogger(log_name)
  logger.setLevel(logging.DEBUG)
  logger.propagate = False

  if logger.hasHandlers():
    logger.handlers.clear()

  fh = logging.FileHandler(log_path)
  fh.setLevel(logging.DEBUG)
  formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
  fh.setFormatter(formatter)
  logger.addHandler(fh)
  for noisy in ["streamlit", "watchdog", "urllib3", "httpx", "asyncio", "PIL","httpcore"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

  return logger