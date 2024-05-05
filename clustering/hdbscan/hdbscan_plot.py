import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
import umap
import os
import time
import sys
sys.path.insert(0, os.path.abspath("../.."))

class Hdbscan():
    def __init__(self, logger, config, file_management):
        self.logger = logger
        self.config = config
        self.file_management = file_management

    def read_files(self, name_embedding_matrix):
        ids = self.file_management.load_pkl("ids.pkl")
        embedding = self.file_management.load_npy(name_embedding_matrix)
        return embedding, ids
    
    def get_embedding(self, embedding_matrix, method):
        self.logger.info(f'Reducing dimensions of embeddings generated with {method} using UMAP.')
        start_proyection = time.time()
        embeddings_reduced = umap.UMAP(n_components=2).fit(embedding_matrix)
        end_proyection = time.time()
        self.logger.info(f'Dimensions reduced using UMAP in {(end_proyection - start_proyection)/60} minutes')
        return embeddings_reduced


    def plot_hdbscan(self, embedding, nodes_labels, method):
        plt.figure(figsize=(10, 6))
        plt.scatter(embedding.embedding_[:, 0], embedding.embedding_[:, 1], alpha=0.4, c=nodes_labels.hdbscan, cmap='viridis')
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(f'HDBSCAN: {method}')
        plt.savefig(self.file_management.join_path(f'hdbscan_{method}.png'))


    def hdbscan_fit(self, embedding_matrix, ids, method):
        start_fit = time.time()
        hdbs_model = hdbscan.HDBSCAN().fit(embedding_matrix)
        end_fit = time.time()
        self.logger.info(f"HDBSCAN over embedding from {str(method)} fit in: {(end_fit - start_fit)/60} minutes")
        hbds_scan_labels = hdbs_model.labels_
        return pd.DataFrame(zip(ids, hbds_scan_labels), columns = ['node_ids','hdbscan'])
    
    def run_hdbscan(self, method, name_npy, name_csv):
        embedding, ids = self.read_files(name_npy)
        embedding_reduced = self.get_embedding(embedding, method)
        nodes_labels = self.hdbscan_fit(embedding, ids, method)
        self.file_management.save_df(name_csv, nodes_labels)
        self.plot_hdbscan(embedding_reduced, nodes_labels, method)
        return nodes_labels
    
    def run(self, method):
        if method == "Custom Embedder":
            return self.run_hdbscan(method, "embedding_matrix.npy", "hdbscan_custom.csv")
        elif method == "Node2Vec":
            return self.run_hdbscan(method, "node2vec_embedding_matrix.npy", "hdbscan_node2vec.csv")
        else:
            self.logger.error("Method not found")