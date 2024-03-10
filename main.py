import argparse
from fake_graph.Fake_graph import GeneratorFakeGraph
from clustering.hdbscan.hdbscan_plot import Hdbscan
from clustering.kmeans.kmeans_plot import Kmeans
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

def calculate_precision(column, indexes, mapping, tmp_mapping, full_nodes, logger, file_management):
    for i in indexes:
        mapping[column] = mapping.get(column, {})
        if isinstance(tmp_mapping.loc[i], np.ndarray):
            mapping[column][tmp_mapping.loc[i][0]] = i
        else:
            mapping[column][tmp_mapping.loc[i]] = i
    
    full_nodes[f"{column}_mapped"] = full_nodes.apply(lambda x: mapping[column].get(int(x[column])), axis=1)

    result = dict()
    
    result_act = list(full_nodes.apply(lambda x: x[f"{column}_mapped"] == x.real_cluster, axis=1).value_counts())
    if len(result_act) == 1:
        if full_nodes.apply(lambda x: x[f"{column}_mapped"] == x.real_cluster, axis=1).value_counts().index[0]:
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

    df = run_fake_graph(logger, file_management)

    executor = Executor(logger, df, config, file_management)
    executor.run_all()

    kmeans_processor = Kmeans(logger, config, file_management)
    nodes_labels_custom_kmeans = kmeans_processor.run("Custom Embedder")
    nodes_labels_node2vec_kmeans = kmeans_processor.run("Node2Vec")

    hdbscan_processor = Hdbscan(logger, config, file_management)
    nodes_labels_custom_hdbscan = hdbscan_processor.run("Custom Embedder")
    nodes_labels_node2vec_hdbscan = hdbscan_processor.run("Node2Vec")

    clusters = dict()
    df.apply(lambda x: set_clusters(x, clusters), axis=1)
    
    dfs = [nodes_labels_custom_hdbscan, nodes_labels_node2vec_hdbscan, nodes_labels_custom_kmeans, nodes_labels_node2vec_kmeans]
    # dfs = [nodes_labels_custom_hdbscan, nodes_labels_custom_kmeans]
    methods = ['hdbscan_custom', 'hdbscan_node2vec', 'kmeans_custom', 'kmeans_node2vec']
    # methods = ['hdbscan_custom', 'kmeans_custom']

    for_df = {}
    for i in range(len(dfs)):
        df = dfs[i]
        method = methods[i]
        df.rename(columns={df.columns[1]: method}, inplace=True)
        df['real_cluster'] = df.apply(lambda x: clusters.get(x.node_ids, 0), axis=1)
        
        mapping = dict()
        tmp_mapping = df.groupby(['real_cluster'])[method].aggregate(mode_without_err)
        indexes = df.real_cluster.value_counts().index

        presicion = calculate_precision(method, indexes, mapping, tmp_mapping, df, logger, file_management)
        for_df = for_df | presicion

    df_result = pd.DataFrame(for_df)
    file_management.save_df("df_result.csv", df_result)

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

    for i in range(0, int(config["RUNNING"]["n"])):
        folder_name = f"{time}_{i}_{hash[:8]}"
        logger = MyLogger(__name__, folder_name, level=args.level, id=hash)
        
        local_config = config
        local_config["KMEANS"] = config[f"KMEANS_{i}"]
        local_config["FAKE_GRAPH"] = config[f"FAKE_GRAPH_{i}"]
        local_config["GENERATOR_V2"] = config[f"GENERATOR_V2_{i}"]
        main(local_config, logger, folder_name)

    # folder_name = f"{time}_{hash[:8]}"
    # logger = MyLogger(__name__, folder_name, level=args.level, id=hash)
    # main(config, logger, folder_name)