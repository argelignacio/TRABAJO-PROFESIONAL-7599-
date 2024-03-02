import argparse
from fake_graph.Fake_graph import GeneratorFakeGraph
from clustering.hdbscan.hdbscan_plot import run_hdbscan
from clustering.kmeans.kmeans_plot import run_kmeans
from logger.logger import MyLogger
from vs_embedders.executor import Executor
import configparser
import uuid
import os
from utils.time import Time
from utils.file_management import FileManagement

def main(level, config):
    hash = str(uuid.uuid4())
    time = Time().datetime()
    folder_name = f"{time}_{hash[:8]}"
    logger = MyLogger(__name__, folder_name, level=level, id=hash)

    file_management = FileManagement(os.path.join(os.getcwd(), "results", folder_name))

    n_clusters = int(config["FAKE_GRAPH"]["n_clusters"])
    nodes_per_cluster = int(config["FAKE_GRAPH"]["nodes_per_cluster"])
    intern_probability = float(config["FAKE_GRAPH"]["intern_probability"])
    n_transactions = int(config["FAKE_GRAPH"]["n_transactions"])

    df = GeneratorFakeGraph.generate_fake_graph_df(logger, n_clusters, nodes_per_cluster, intern_probability, n_transactions)
    file_management.save_df("fake_graph.csv", df)

    executor = Executor(logger, df, config, file_management)
    executor.run_all()

    run_kmeans(logger, config, file_management)
    run_hdbscan(logger, config, file_management)

if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read('./config/config.ini')
    
    parser = argparse.ArgumentParser(description='Script de ejemplo con registro')

    choices = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    message = 'Nivel de registro DEBUG, INFO, WARNING,W ERROR, CRITICAL'
    parser.add_argument('--level', choices=choices, default='INFO', help=message)
    args = parser.parse_args()
    main(args.level, config)