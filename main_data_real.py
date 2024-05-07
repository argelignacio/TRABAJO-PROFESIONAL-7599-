import argparse
from clustering.hdbscan.hdbscan_plot import Hdbscan
from clustering.kmeans.kmeans_plot import Kmeans
from logger.logger import MyLogger
from clustering.embedders.processing_frames import ProcessingFrames
import configparser
import uuid
import os
from utils.time import Time
from utils.file_management import FileManagement
from clustering.embedders.processing_frames import ProcessingFrames

def main(config, logger, folder_name):
    file_management = FileManagement(os.path.join(os.getcwd(), "results", folder_name))

    files = [
        "../datos/2023/July/2023-07-02.csv"
    ]

    processing_frames = ProcessingFrames.build_from_files(files, logger)
    embedding_matrix, ids = processing_frames.pipeline(config)

    file_management.save_npy("embedding_matrix.npy", embedding_matrix[0])
    logger.debug(f"Saved file embedding_matrix.npy")
    file_management.save_pkl("ids.pkl", ids)
    logger.debug(f"Saved file ids.pkl")

    kmeans_processor = Kmeans(logger, config, file_management)
    kmeans_processor.run("Custom Embedder")

    hdbscan_processor = Hdbscan(logger, config, file_management)
    hdbscan_processor.run("Custom Embedder")

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('./config/config.ini')

    parser = argparse.ArgumentParser(description='Script de ejemplo con registro')

    choices = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    message = 'Nivel de registro DEBUG, INFO, WARNING, ERROR, CRITICAL'
    parser.add_argument('--level', choices=choices, default='INFO', help=message)
    args = parser.parse_args()

    hash = str(uuid.uuid4())
    time = Time().datetime()

    folder_name = f"{time}_{hash[:8]}"
    logger = MyLogger(__name__, folder_name, level=args.level, id=hash)
    main(config, logger, folder_name)