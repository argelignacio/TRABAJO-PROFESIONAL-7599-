from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import umap
import os
import time 
import sys
sys.path.insert(0, os.path.abspath("../.."))

def read_files(file_management):
    ids = file_management.load_pkl("ids.pkl")
    embedding_matrix_custom = file_management.load_npy("embedding_matrix.npy")
    embedding_matrix_node2vec = file_management.load_npy("node2vec_embedding_matrix.npy")
    return embedding_matrix_custom, embedding_matrix_node2vec, ids


def plot_elbow(embedding_matrix_custom, embedding_matrix_node2vec, file_management):
    distortions1 = []
    distortions2 = []
    K = range(3, 16)

    for k in K:
        k_cluster1 = KMeans(n_clusters=k, max_iter=500, random_state=3425).fit(embedding_matrix_custom)
        k_cluster1.fit(embedding_matrix_custom)
        distortions1.append(k_cluster1.inertia_)

        k_cluster2 = KMeans(n_clusters=k, max_iter=500, random_state=3425).fit(embedding_matrix_node2vec)
        k_cluster2.fit(embedding_matrix_node2vec)
        distortions2.append(k_cluster2.inertia_)

    _, axs = plt.subplots(1, 2, figsize=(16, 6))

    axs[0].plot(K, distortions1, 'bx-')
    axs[0].set_xlabel('k')
    axs[0].set_ylabel('Distortion')
    axs[0].set_title('The Elbow Method for Embedding Custom')

    axs[1].plot(K, distortions2, 'bx-')
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('Distortion')
    axs[1].set_title('The Elbow Method for Embedding Node2Vec')

    plt.savefig(file_management.join_path('elbow_method.png'))


def plot_kmeans(embedding_custom, embedding_n2v, nodes_labels_custom, nodes_labels_n2v, file_management):
    _, axs = plt.subplots(1, 2, figsize=(20, 6))

    axs[0].scatter(embedding_custom.embedding_[:, 0], embedding_custom.embedding_[:, 1], alpha=0.4, c=nodes_labels_custom.kmeans, cmap='viridis')
    axs[0].set_aspect('equal', 'datalim')
    axs[0].set_title('KMEANS: Embedding custom')

    length = len(nodes_labels_custom)
    embedding_n2v_trimmed = embedding_n2v.embedding_[:length, :]
    axs[1].scatter(embedding_n2v_trimmed[:, 0], embedding_n2v_trimmed[:, 1], alpha=0.4, c=nodes_labels_n2v.kmeans, cmap='viridis')
    axs[1].set_aspect('equal', 'datalim')
    axs[1].set_title('KMEANS: Embedding node2vec')

    plt.savefig(file_management.join_path('kmeans.png'))


def kmeans_fit(logger, embedding_matrix, ids, n_clusters, init, n_init, method, random_state=3425):
    start_fit = time.time()
    kmeans_cluster = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, random_state=random_state).fit(embedding_matrix)
    end_fit = time.time()
    logger.info(f"KMeans over embedding from {str(method)} fitted with n_clusters: {str(n_clusters)} init: {str(init)} n_init: {str(n_init)} random_state:  {str(random_state)} fit in: {(end_fit - start_fit)/60} minutes")
    kmeans_labels = kmeans_cluster.labels_
    return pd.DataFrame(zip(ids, kmeans_labels), columns = ['node_ids','kmeans'])

def get_embedding(embedding_matrix, logger, method):
    logger.info(f'Reducing dimensions of embeddings generated with {method} using UMAP.')
    start_proyection = time.time()
    embeddings_reduced = umap.UMAP(n_components=2).fit(embedding_matrix)
    end_proyection = time.time()
    logger.info(f'Dimensions reduced using UMAP in {(end_proyection - start_proyection)/60} minutes')
    return embeddings_reduced


def run_kmeans(logger, config, file_management):
    n_init = int(config["KMEANS"]["n_clusters"])
    n_clusters = int(config["KMEANS"]["n_init"])
    embedding_matrix_custom, embedding_matrix_node2vec, ids = read_files(file_management)
    embedding_custom = get_embedding(embedding_matrix_custom, logger, "Custom Embedder")
    embedding_node2vec = get_embedding(embedding_matrix_node2vec, logger, "Node2Vec")
    nodes_labels_custom = kmeans_fit(logger, embedding_matrix_custom, ids, n_clusters, 'k-means++', n_init, 'Custom Embedder')
    nodes_labels_node2vec = kmeans_fit(logger, embedding_matrix_node2vec, ids, n_clusters, 'k-means++', n_init, 'Node2Vec')
    file_management.save_df('kmeans_custom_csv', nodes_labels_custom)
    file_management.save_df('kmeans_n2v.csv', nodes_labels_node2vec)
    plot_elbow(embedding_matrix_custom, embedding_matrix_node2vec, file_management)
    plot_kmeans(embedding_custom, embedding_node2vec, nodes_labels_custom, nodes_labels_node2vec, file_management)