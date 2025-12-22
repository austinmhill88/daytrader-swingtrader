from loguru import logger
import os

def setup_logging(logs_dir: str):
    os.makedirs(logs_dir, exist_ok=True)
    logger.remove()
    logger.add(os.path.join(logs_dir, "runtime.log"), rotation="1 day", retention="30 days", level="INFO")
    logger.add(os.path.join(logs_dir, "errors.log"), rotation="1 day", retention="60 days", level="ERROR")
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    return logger