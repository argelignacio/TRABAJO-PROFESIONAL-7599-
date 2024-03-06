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

def calculate_precision(columns, indexes, mapping, tmp_mapping, full_nodes, logger, file_management):
    for col in columns:
        for i in indexes:
            mapping[col] = mapping.get(col, {})
            if isinstance(tmp_mapping.loc[i, col], np.ndarray):
                mapping[col][tmp_mapping.loc[i, col][0]] = i
            else:
                mapping[col][tmp_mapping.loc[i, col]] = i
    
    for col in columns:
        full_nodes[f"{col}_mapped"] = full_nodes.apply(lambda x: mapping[col].get(int(x[col])), axis=1)

    result = dict()
    for col in ['hdbscan_custom_mapped', 'hdbscan_node2vec_mapped', 'kmeans_custom_mapped', 'kmeans_node2vec_mapped']:
        result_act = list(full_nodes.apply(lambda x: x[col] == x.real_cluster, axis=1).value_counts())
        if len(result_act) == 1:
            logger.info(f"Precision_{col}: 1")
            result[f"Precision_{col}"] = result.get(f"Precision_{col}", 1)
        else:
            precision = result.get(f"Precision_{col}", (int(result_act[1]) / (int(result_act[0])+int(result_act[1]))))
            logger.info(f"Precision_{col}: {precision}")
            result[f"Precision_{col}"] = precision

    for_df = {}
    for key in result.keys():
        for_df[key] = [result[key]]

    df_result = pd.DataFrame(for_df)
    file_management.save_df("df_result.csv", df_result)

def main(config, logger, folder_name):
    file_management = FileManagement(os.path.join(os.getcwd(), "results", folder_name))

    df = run_fake_graph(logger, file_management)

    executor = Executor(logger, df, config, file_management)
    executor.run_all()

    kmeans_processor = Kmeans(logger, config, file_management)
    nodes_labels_custom_kmeans = kmeans_processor.run("Custom Embedder")
    nodes_labels_node2vec_kmeans = kmeans_processor.run("Node2Vec")

    nodes_labels_custom_hdbscan, nodes_labels_node2vec_hdbscan = run_hdbscan(logger, config, file_management)

    full_nodes = nodes_labels_node2vec_hdbscan\
        .merge(nodes_labels_custom_hdbscan, left_on='node_ids', right_on='node_ids', suffixes=('_node2vec', '_custom'))\
        .merge(nodes_labels_custom_kmeans, left_on='node_ids', right_on='node_ids')\
        .merge(nodes_labels_node2vec_kmeans, left_on='node_ids', right_on='node_ids', suffixes=('_custom', '_node2vec'))

    clusters = dict()

    df.apply(lambda x: set_clusters(x, clusters), axis=1)
    full_nodes['real_cluster'] = full_nodes.apply(lambda x: clusters.get(x.node_ids), axis=1)

    mapping = dict()
    columns = ['hdbscan_custom', 'hdbscan_node2vec', 'kmeans_custom', 'kmeans_node2vec']

    tmp_mapping = full_nodes.groupby(['real_cluster'])[columns].aggregate(mode_without_err)
    indexes = full_nodes.real_cluster.value_counts().index

    full_nodes["real_cluster_modified"] = full_nodes.apply(lambda x: mapping.get(x.real_cluster), axis=1)

    calculate_precision(columns, indexes, mapping, tmp_mapping, full_nodes, logger, file_management)

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

    # for i in range(0, int(config["RUNNING"]["n"])):
    #     folder_name = f"{time}_{i}_{hash[:8]}"
    #     logger = MyLogger(__name__, folder_name, level=args.level, id=hash)
        
    #     local_config = config
    #     local_config["KMEANS"] = config[f"KMEANS_{i}"]
    #     local_config["FAKE_GRAPH"] = config[f"FAKE_GRAPH_{i}"]
    #     main(local_config, logger, folder_name)

    folder_name = f"{time}_{hash[:8]}"
    logger = MyLogger(__name__, folder_name, level=args.level, id=hash)
    main(config, logger, folder_name)