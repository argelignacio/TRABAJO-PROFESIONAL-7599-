import logging
from logging.handlers import RotatingFileHandler
import os
from utils.time import Time

class MyLogger:
    def __init__(self, name, log_folder, log_file=None, level=logging.DEBUG, id = 0):
        self.id = id
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        if log_file is None:
            folder_path = os.path.join(os.getcwd(), "results", log_folder)
            os.makedirs(folder_path, exist_ok=True)
            file_name = f"logs.log"
            log_file = os.path.join(folder_path, file_name)
        
        self.log_path = folder_path

        # Rotating file handler
        file_handler = RotatingFileHandler(log_file, mode='a', maxBytes=10*1024*1024, backupCount=5)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def get_log_path(self):
        return self.log_path

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