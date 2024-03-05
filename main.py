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


def run_fake_graph(logger, file_management):
    n_clusters = int(config["FAKE_GRAPH"]["n_clusters"])
    nodes_per_cluster = int(config["FAKE_GRAPH"]["nodes_per_cluster"])
    intern_probability = float(config["FAKE_GRAPH"]["intern_probability"])
    n_transactions = int(config["FAKE_GRAPH"]["n_transactions"])

    df = GeneratorFakeGraph.generate_fake_graph_df(logger, n_clusters, nodes_per_cluster, intern_probability, n_transactions)
    file_management.save_df("fake_graph.csv", df)
    return df

def set_clusters(row, clusters):
    clusters[row.from_address] = row.cluster_from
    clusters[row.to_address] = row.cluster_to

def main(level, config):
    hash = str(uuid.uuid4())
    time = Time().datetime()
    folder_name = f"{time}_{hash[:8]}"
    logger = MyLogger(__name__, folder_name, level=level, id=hash)

    file_management = FileManagement(os.path.join(os.getcwd(), "results", folder_name))

    df = run_fake_graph(logger, file_management)

    executor = Executor(logger, df, config, file_management)
    executor.run_all()

    nodes_labels_custom_kmeans, nodes_labels_node2vec_kmeans = run_kmeans(logger, config, file_management)
    nodes_labels_custom_hdbscan, nodes_labels_node2vec_hdbscan = run_hdbscan(logger, config, file_management)

    full_nodes = nodes_labels_node2vec_hdbscan\
        .merge(nodes_labels_custom_hdbscan, left_on='node_ids', right_on='node_ids', suffixes=('_custom', '_node2vec'))\
        .merge(nodes_labels_custom_kmeans, left_on='node_ids', right_on='node_ids')\
        .merge(nodes_labels_node2vec_kmeans, left_on='node_ids', right_on='node_ids', suffixes=('_custom', '_node2vec'))
    
    clusters = dict()

    df.apply(lambda x: set_clusters(x, clusters), axis=1)
    full_nodes['real_cluster'] = full_nodes.apply(lambda x: int(clusters.get(x.node_ids)), axis=1)
    mapping = full_nodes.groupby(['real_cluster'])[['hdbscan_custom', 'hdbscan_node2vec', 'kmeans_custom', 'kmeans_node2vec']]\
        .mean()\
        .apply(lambda x: round(x), axis=1)

    full_nodes["real_cluster_modified"] = full_nodes.apply(lambda x: mapping.get(x.real_cluster), axis=1)

    result = dict()    
    for col in ['hdbscan_custom', 'hdbscan_node2vec', 'kmeans_custom', 'kmeans_node2vec']:
        result = list(full_nodes.apply(lambda x: x[col] == x.real_cluster_modified, axis=1).value_counts())
        result.insert(f"Precision_{col}", list[0] / (list[0]+list[1]))

    print(result)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('./config/config.ini')

    parser = argparse.ArgumentParser(description='Script de ejemplo con registro')

    choices = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    message = 'Nivel de registro DEBUG, INFO, WARNING, ERROR, CRITICAL'
    parser.add_argument('--level', choices=choices, default='INFO', help=message)
    args = parser.parse_args()
    main(args.level, config)