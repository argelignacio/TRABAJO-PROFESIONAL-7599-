import argparse
from fake_graph.Fake_graph import GeneratorFakeGraph
from clustering.hdbscan.hdbscan_plot import Hdbscan
from clustering.kmeans.kmeans_plot import Kmeans
from logger.logger import MyLogger
from clustering.embedders.processing_frames import pipeline
import configparser
import uuid
import os
from utils.time import Time
from utils.file_management import FileManagement
import pandas as pd
import numpy as np

def run_fake_graph(logger, file_management):
    n_clusters = int(config["KMEANS"]["n_clusters"])
    nodes_per_cluster = int(config["FAKE_GRAPH"]["nodes_per_cluster"])
    intern_probability = float(config["FAKE_GRAPH"]["intern_probability"])
    n_transactions = int(config["FAKE_GRAPH"]["n_transactions"])

    df = GeneratorFakeGraph.generate_fake_graph_df(logger, n_clusters, nodes_per_cluster, intern_probability, n_transactions)
    file_management.save_df("fake_graph.csv", df)
    return df

def set_clusters(row, clusters):
    clusters[row.from_address] = row.cluster_from
    clusters[row.to_address] = row.cluster_to

def mode_without_err(x):
    mode = pd.Series.mode(x)
    if len(mode) > 1:
        return mode[0]
    return mode

def calculate_precision(column, indexes, mapping, tmp_mapping, full_nodes, logger, file_management):
    for i in indexes:
        mapping[column] = mapping.get(column, {})
        if isinstance(tmp_mapping.loc[i], np.ndarray):
            mapping[column][tmp_mapping.loc[i][0]] = i
        else:
            mapping[column][tmp_mapping.loc[i]] = i
    
    full_nodes[f"{column}_mapped"] = full_nodes.apply(lambda x: mapping[column].get(int(x[column])), axis=1)

    result = dict()
    
    result_act = list(full_nodes.apply(lambda x: x[column] == x.real_cluster, axis=1).value_counts())
    if len(result_act) == 1:
        if full_nodes.apply(lambda x: x[column] == x.real_cluster, axis=1).value_counts().index[0]:
            logger.info(f"Precision_{column}: 1")
            result[f"Precision_{column}"] = result.get(f"Precision_{column}", 1)
        else:
            logger.info(f"Precision_{column}: 0")
            result[f"Precision_{column}"] = result.get(f"Precision_{column}", 0)
    else:
        precision = result.get(f"Precision_{column}", (int(result_act[1]) / (int(result_act[0])+int(result_act[1]))))
        logger.info(f"Precision_{column}: {precision}")
        result[f"Precision_{column}"] = precision

    for_df = {}
    for key in result.keys():
        for_df[key] = [result[key]]

    return for_df

def main(config, logger, folder_name):
    file_management = FileManagement(os.path.join(os.getcwd(), "results", folder_name))
    
    # files = ["../datos/2023/July/2023-07-02.csv"]

    # embedding_matrix, ids = pipeline(files, logger, config)
    # file_management.save_npy("embedding_matrix.npy", embedding_matrix[0])
    # logger.debug(f"Saved file embedding_matrix.npy")
    # file_management.save_pkl("ids.pkl", ids)
    # logger.debug(f"Saved file ids.pkl")

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

    # hash = str(uuid.uuid4())
    # time = Time().datetime()

    hash = "2e78502a-5ebf-4768-9cf3-ee719df28471"
    time = "2024-03-09 17:54:52"

    folder_name = f"{time}_{hash[:8]}"
    logger = MyLogger(__name__, folder_name, level=args.level, id=hash)
    main(config, logger, folder_name)