import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import umap
import os
from datetime import datetime
import time
import sys
sys.path.insert(0, os.path.abspath("../.."))


def read_files(file_management):
    ids = file_management.load_pkl("ids.pkl")
    embedding_matrix_custom = file_management.load_npy("embedding_matrix.npy")
    embedding_matrix_node2vec = file_management.load_npy("node2vec_embedding_matrix.npy")
    return embedding_matrix_custom, embedding_matrix_node2vec, ids


def plot_hdbscan(embedding_custom, embedding_n2v, nodes_labels_custom, nodes_labels_n2v, file_management):
    _, axs = plt.subplots(1, 2, figsize=(20, 6))

    axs[0].scatter(embedding_custom.embedding_[:, 0], embedding_custom.embedding_[:, 1], alpha=0.4, c=nodes_labels_custom.hdbscan, cmap='viridis')
    axs[0].set_aspect('equal', 'datalim')
    axs[0].set_title('HDBSCAN: Embedding custom')

    length = len(nodes_labels_custom)
    embedding_n2v_trimmed = embedding_n2v.embedding_[:length, :]
    axs[1].scatter(embedding_n2v_trimmed[:, 0], embedding_n2v_trimmed[:, 1], alpha=0.4, c=nodes_labels_n2v.hdbscan, cmap='viridis')
    axs[1].set_aspect('equal', 'datalim')
    axs[1].set_title('HDBSCAN: Embedding node2vec')

    plt.savefig(file_management.join_path('hdbscan.png'))


def hdbscan_fit(logger, embedding_matrix, ids, method):
    start_fit = time.time()
    hdbs_model = hdbscan.HDBSCAN().fit(embedding_matrix)
    end_fit = time.time()
    logger.info(f"HDBSCAN over embedding from {str(method)} fit in: {(end_fit - start_fit)/60} minutes")
    hbds_scan_labels = hdbs_model.labels_
    return pd.DataFrame(zip(ids, hbds_scan_labels), columns = ['node_ids','hdbscan'])

def get_embedding(embedding_matrix, logger, method):
    start_proyection = time.time()
    embeddings_reduced = umap.UMAP(n_components=2).fit(embedding_matrix)
    end_proyection = time.time()
    logger.info(f'Dimensions reduced with {method} using UMAP in {(end_proyection - start_proyection)/60} minutes')
    return embeddings_reduced

def run_hdbscan(logger, _config, file_management):
    embedding_matrix_custom, embedding_matrix_node2vec, ids = read_files(file_management)
    embedding_custom = get_embedding(embedding_matrix_custom, logger, "Custom Embedder")
    embedding_node2vec = get_embedding(embedding_matrix_node2vec, logger, "Node2Vec")
    nodes_labels_custom = hdbscan_fit(logger, embedding_matrix_custom, ids, 'Custom Embedder')
    nodes_labels_node2vec = hdbscan_fit(logger, embedding_matrix_node2vec, ids, 'Node2Vec')

    file_management.save_df('hdbscan_custom.csv', nodes_labels_custom)
    file_management.save_df('hdbscan_n2v.csv', nodes_labels_node2vec)
    plot_hdbscan(embedding_custom, embedding_node2vec, nodes_labels_custom, nodes_labels_node2vec, file_management)
    return nodes_labels_custom, nodes_labels_node2vec