import argparse
from fake_graph.Fake_graph import GeneratorFakeGraph
from clustering.hdbscan.hdbscan_plot import run_hdbscan
from clustering.kmeans.kmeans_plot import run_kmeans
from logger.logger import MyLogger
from vs_embedders.executor import Executor

def main(level):
    logger = MyLogger(__name__, level=level, id=1)
    n_clusters = 6
    nodes_per_cluster = 300
    intern_probability = 0.85
    n_transactions = 3000
    df = GeneratorFakeGraph.generate_fake_graph_df(logger, n_clusters, nodes_per_cluster, intern_probability, n_transactions)
    df.to_csv("fake_graph.csv", index=False)
    executor = Executor(logger, df)
    executor.run_all()

    run_kmeans(logger)
    run_hdbscan()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script de ejemplo con registro')

    choices = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    message = 'Nivel de registro (DEBUG, INFO, WARNING, ERROR, CRITICAL'
    parser.add_argument('--level', choices=choices, default='INFO', help=message)
    args = parser.parse_args()

    main(args.level)