from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import umap
import os
import time
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.abspath("../.."))

class Kmeans():
    def __init__(self, logger, config, file_management):
        self.logger = logger
        self.config = config
        self.file_management = file_management
        self.n_init = int(config["KMEANS"]["n_init"])
        self.n_clusters = int(config["KMEANS"]["n_clusters"])
        self.init = "k-means++"

    def read_files(self, name_embedding_matrix):
        ids = self.file_management.load_pkl("ids.pkl")
        embedding = self.file_management.load_npy(name_embedding_matrix)
        return embedding, ids


    def plot_elbow(self, embedding_matrix, method):
        distortions = []
        K = range(1, self.n_clusters, 10)

        for k in tqdm(K):
            self.logger.info(f"n_iter: {k}")
            k_cluster1 = KMeans(n_clusters=k, init=self.init, n_init=self.n_init, random_state=3425).fit(embedding_matrix)
            k_cluster1.fit(embedding_matrix)
            distortions.append(k_cluster1.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title(f'The Elbow Method for {method}')
        plt.savefig(self.file_management.join_path(f'elbow_method_{method}.png'))


    def plot_kmeans(self, embedding_custom, nodes_labels, method, ids):
        plt.figure(figsize=(10, 6))
        plt.scatter(embedding_custom.embedding_[:, 0], embedding_custom.embedding_[:, 1], alpha=0.4, c=nodes_labels.kmeans, cmap='viridis')
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(f'KMEANS: {method}')
        plt.savefig(self.file_management.join_path(f'kmeans_{method}.png'))


    def kmeans_fit(self, embedding_matrix, ids, method, random_state=3425):
        start_fit = time.time()
        kmeans_cluster = KMeans(n_clusters=self.n_clusters, init=self.init, n_init=self.n_init, random_state=random_state).fit(embedding_matrix)
        end_fit = time.time()
        self.logger.info(f"KMeans over embedding from {str(method)} fitted with n_clusters: {str(self.n_clusters)} init: {str(self.init)} n_init: {str(self.n_init)} random_state:  {str(random_state)} fit in: {(end_fit - start_fit)/60} minutes")
        kmeans_labels = kmeans_cluster.labels_
        return pd.DataFrame(zip(ids, kmeans_labels), columns = ['node_ids','kmeans'])

    def get_embedding(self, embedding_matrix, method):
        self.logger.info(f'Reducing dimensions of embeddings generated with {method} using UMAP.')
        start_proyection = time.time()
        embeddings_reduced = umap.UMAP(n_components=2).fit(embedding_matrix)
        end_proyection = time.time()
        self.logger.info(f'Dimensions reduced using UMAP in {(end_proyection - start_proyection)/60} minutes')
        return embeddings_reduced
    
    def run_kmeans(self, method, name_npy, name_csv):
        embedding, ids = self.read_files(name_npy)
        embedding_reduced = self.get_embedding(embedding, method)
        # self.plot_elbow(embedding, method)
        nodes_labels = self.kmeans_fit(embedding, ids, method)
        self.file_management.save_df(name_csv, nodes_labels)
        self.plot_kmeans(embedding_reduced, nodes_labels, method, ids)
        return nodes_labels
    
    def run(self, method):
        if method == "Custom Embedder":
            return self.run_kmeans(method, "embedding_matrix.npy", "kmeans_custom.csv")
        elif method == "Node2Vec":
            return self.run_kmeans(method, "node2vec_embedding_matrix.npy", "kmeans_node2vec.csv")
        else:
            self.logger.error("Method not found")
