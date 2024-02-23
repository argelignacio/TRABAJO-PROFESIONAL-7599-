import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

class MyLogger:
    def __init__(self, name, log_file=None, level=logging.DEBUG, id = 0):
        self.id = id
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        if log_file is None:
            project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
            log_dir = os.path.join(project_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d") + ".log")

        # Rotating file handler
        file_handler = RotatingFileHandler(log_file, mode='a', maxBytes=10*1024*1024, backupCount=5)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def debug(self, message):
        self.logger.debug(f"{self.id} | {message}")

    def info(self, message):
        self.logger.info(f"{self.id} | {message}")

    def warning(self, message):
        self.logger.warning(f"{self.id} | {message}")

    def error(self, message):
        self.logger.error(f"{self.id} | {message}")

    def critical(self, message):
        self.logger.critical(f"{self.id} | {message}")