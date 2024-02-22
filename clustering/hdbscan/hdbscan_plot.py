import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import umap
import os
from datetime import datetime
import sys
sys.path.insert(0, os.path.abspath("../.."))
from logger.logger import MyLogger

logger = MyLogger(__name__)

def read_files():
    with open("vs_embedders/ids.pkl", "rb") as file:
        ids = pickle.load(file)
    embedding_matrix_custom = np.load("vs_embedders/custom_embedding_matrix.npy", allow_pickle=True)
    embedding_matrix_node2vec = np.load("vs_embedders/node2vec_embedding_matrix.npy", allow_pickle=True)
    return embedding_matrix_custom, embedding_matrix_node2vec, ids


def plot_hdbscan(embedding_custom, embedding_n2v, nodes_labels_custom, nodes_labels_n2v):
    _, axs = plt.subplots(1, 2, figsize=(20, 6))

    axs[0].scatter(embedding_custom.embedding_[:, 0], embedding_custom.embedding_[:, 1], alpha=0.4, c=nodes_labels_custom.hdbscan, cmap='viridis')
    axs[0].set_aspect('equal', 'datalim')
    axs[0].set_title('HDBSCAN: Embedding custom')

    length = len(nodes_labels_custom)
    embedding_n2v_trimmed = embedding_n2v.embedding_[:length, :]
    axs[1].scatter(embedding_n2v_trimmed[:, 0], embedding_n2v_trimmed[:, 1], alpha=0.4, c=nodes_labels_n2v.hdbscan, cmap='viridis')
    axs[1].set_aspect('equal', 'datalim')
    axs[1].set_title('HDBSCAN: Embedding node2vec')

    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'hdbscan_{current_time}.png'
    parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    plots_directory = os.path.join(parent_directory, "plots")

    if not os.path.exists(plots_directory):
        os.makedirs(plots_directory)

    plt.savefig(os.path.join(plots_directory, filename))


def hdbscan_fit(embedding_matrix, ids):
    hdbs_model = hdbscan.HDBSCAN().fit(embedding_matrix)
    hbds_scan_labels = hdbs_model.labels_
    return pd.DataFrame(zip(ids, hbds_scan_labels), columns = ['node_ids','hdbscan'])


def get_embedding(embedding_matrix):
    return umap.UMAP(n_components=2).fit(embedding_matrix)


def run_hdbscan():
    embedding_matrix_custom, embedding_matrix_node2vec, ids = read_files()
    embedding_custom = get_embedding(embedding_matrix_custom)
    embedding_node2vec = get_embedding(embedding_matrix_node2vec)
    nodes_labels_custom = hdbscan_fit(embedding_matrix_custom, ids)
    nodes_labels_node2vec = hdbscan_fit(embedding_matrix_node2vec, ids)
    plot_hdbscan(embedding_custom, embedding_node2vec, nodes_labels_custom, nodes_labels_node2vec)