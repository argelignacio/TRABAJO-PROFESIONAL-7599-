import os, sys
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import pickle as pkl

sys.path.insert(0, os.path.abspath(".."))
from clustering.embedders.processing_frames import pipeline_v2

class Executor:
    def __init__(self, logger, df, config, folder_name) -> None:
        self.logger = logger
        self.df = df
        self.config = config
        self.folder_name = folder_name

    def _custom_embedding(self):
        embedding_matrix, ids = pipeline_v2(self.df, self.logger, self.config)
        folder_path = os.path.join(os.getcwd(), "results", self.folder_name)
        os.makedirs(folder_path, exist_ok=True)

        np.array(embedding_matrix[0]).dump(os.path.join(folder_path, f"embedding_matrix.npy"))
        self.logger.debug(f"Saved file embedding_matrix.npy")

        pkl.dump(ids, open(os.path.join(folder_path, f"ids.pkl"), "wb"))
        self.logger.debug(f"Saved file ids.pkl")

    def _node2vec_embedding(self, dimensions=128, walk_length=16, num_walks=100, workers=2, window=10, min_count=1, batch_words=4):
        G = nx.from_pandas_edgelist(self.df, 'from_address', 'to_address')
        node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
        self.logger.info("Node2Vec model created with dimensions: " + str(dimensions) + " walk_length: " + str(walk_length) + " num_walks: " + str(num_walks) + " workers: " + str(workers))
        
        model_2 = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
        self.logger.info("Node2Vec model fitted with window: " + str(window) + " min_count: " + str(min_count) + " batch_words: " + str(batch_words))
        
        folder_path = os.path.join(os.getcwd(), "results", self.folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        np.array(model_2.wv.vectors).dump(os.path.join(folder_path, f"node2vec_embedding_matrix.npy"))
        self.logger.debug(f"Saved file node2vec_embedding_matrix.npy")

    def run_all(self):
        self._custom_embedding()
        self._node2vec_embedding()



