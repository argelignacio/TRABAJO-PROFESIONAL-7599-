import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
import umap
import os
import time
import sys
import matplotlib.colors as colors
import numpy as np
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


    def plot_hdbscan(self, embedding, nodes_labels, method, custom_embedding_matrix=None):
        plt.figure(figsize=(10, 6))

        cmap_colors = ['gray'] + [colors.to_hex(plt.cm.viridis(i)) for i in range(len(nodes_labels.hdbscan) - 1)]

        cmap = colors.ListedColormap(cmap_colors)

        plt.scatter(embedding.embedding_[:, 0], embedding.embedding_[:, 1],
                    alpha=0.4, c=nodes_labels.hdbscan, cmap=cmap)

        plt.gca().set_aspect('equal', 'datalim')
        plt.title(f'HDBSCAN: {method}')
        if custom_embedding_matrix:
            plt.savefig(self.file_management.join_path(f'hdbscan_{custom_embedding_matrix}.png'))
        else:
            plt.savefig(self.file_management.join_path(f'hdbscan_{method}.png'))


    def hdbscan_fit(self, embedding_matrix, ids, method):
        start_fit = time.time()
        hdbs_model = hdbscan.HDBSCAN(min_samples=1, cluster_selection_epsilon=0.01).fit(embedding_matrix)
        end_fit = time.time()
        self.logger.info(f"HDBSCAN over embedding from {str(method)} fit in: {(end_fit - start_fit)/60} minutes")
        hbds_scan_labels = hdbs_model.labels_
        return pd.DataFrame(zip(ids, hbds_scan_labels), columns = ['node_ids','hdbscan'])
    
    def run_hdbscan(self, method, name_npy, name_csv):
        embedding, ids = self.read_files(name_npy)
        embedding_reduced = self.get_embedding(embedding, method)
        if method == "Umap":
            nodes_labels = self.hdbscan_fit(embedding_reduced.embedding_, ids, method)
        else:
            nodes_labels = self.hdbscan_fit(embedding, ids, method)
        self.file_management.save_df(name_csv, nodes_labels)
        if method == "Umap":
            self.plot_hdbscan(embedding_reduced, nodes_labels, method)
        else:
            self.plot_hdbscan(embedding_reduced, nodes_labels, method, custom_embedding_matrix=name_npy.split('.')[0])
        return nodes_labels
    
    def _calculate_embedding_sizes(self):
        model_config = self.config["MODEL_V2"]
        current_embedding_size = int(model_config["embedding_dim"])
        embedding_sizes = []
        while current_embedding_size >= 2:
            embedding_sizes.append(current_embedding_size)
            current_embedding_size = current_embedding_size // 2
        return embedding_sizes


    def run(self, method):
        if method == "Custom Embedder":
            dfs = []
            for embedding_size in self._calculate_embedding_sizes():
                dfs.append(self.run_hdbscan(method, f"embedding_matrix_{embedding_size}.npy", f"hdbscan_custom_{embedding_size}.csv"))
            return dfs
        if method == "Umap":
            embedding_size = int(self.config["MODEL_V2"]["embedding_dim"])
            return self.run_hdbscan(method, f"embedding_matrix_{embedding_size}.npy", f"hdbscan_custom_{embedding_size}.csv")
        elif method == "Node2Vec":
            return self.run_hdbscan(method, "node2vec_embedding_matrix.npy", "hdbscan_node2vec.csv")
        else:
            self.logger.error("Method not found")