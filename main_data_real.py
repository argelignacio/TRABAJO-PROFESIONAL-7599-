import argparse
from clustering.hdbscan.hdbscan_plot import Hdbscan
from logger.logger import MyLogger
from clustering.embedders.processing_frames import ProcessingFrames
import configparser
import uuid
import os
from utils.time import Time
from utils.file_management import FileManagement
from clustering.embedders.processing_frames import ProcessingFrames

def start(config, logger, folder_name, pending_model=False):
    file_management = FileManagement(os.path.join(os.getcwd(), "results", folder_name))

    files = [
        "../datos/2023/July/2023-07-01.csv"
    ]

    processing_frames = ProcessingFrames.build_from_files(files, logger)
    embedding_matrix, ids = processing_frames.pipeline(config, pending_model)

    file_management.save_npy("embedding_matrix.npy", embedding_matrix[0])
    logger.debug(f"Saved file embedding_matrix.npy")
    file_management.save_pkl("ids.pkl", ids)
    logger.debug(f"Saved file ids.pkl")

    hdbscan_processor = Hdbscan(logger, config, file_management)
    hdbscan_processor.run("Custom Embedder")

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('./config/config.ini')

    parser = argparse.ArgumentParser(description='Script de ejemplo con registro')

    choices = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    message = 'Nivel de registro DEBUG, INFO, WARNING, ERROR, CRITICAL'
    parser.add_argument('--level', choices=choices, default='INFO', help=message)
    parser.add_argument('--file', default=None, help=message)
    parser.add_argument('--hash', default=None, help=message)
    args = parser.parse_args()

    if not args.file:
        hash = str(uuid.uuid4())
        time = Time().datetime()

        folder_name = f"{time}_{hash[:8]}"
        logger = MyLogger(__name__, folder_name, level=args.level, id=hash)
        start(config, logger, folder_name)
    else:
        logger = MyLogger(__name__, args.file, level=args.level, id=args.hash)
        start(config, logger, args.file, pending_model=True)