import os, sys
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import pickle as pkl

sys.path.insert(0, os.path.abspath(".."))
from clustering.embedders.processing_frames import pipeline_v2

class Executor:
    def __init__(self, logger, df) -> None:
        self.logger = logger
        self.df = df

    def _custom_embedding(self):
        embedding_matrix, ids = pipeline_v2(self.df, self.logger)
        np.array(embedding_matrix[0]).dump("embedding_matrix.npy")
        self.logger.debug("Save file embedding_matrix.npy")
        pkl.dump(ids, open("ids.pkl", "wb"))
        self.logger.debug("Save file ids.pkl")

    def _node2vec_embedding(self, dimensions=128, walk_length=16, num_walks=100, workers=2, window=10, min_count=1, batch_words=4):
        G = nx.from_pandas_edgelist(self.df, 'from_address', 'to_address')
        node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
        self.logger.info("Node2Vec model created with dimensions: " + str(dimensions) + " walk_length: " + str(walk_length) + " num_walks: " + str(num_walks) + " workers: " + str(workers))
        model_2 = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
        self.logger.info("Node2Vec model fitted with window: " + str(window) + " min_count: " + str(min_count) + " batch_words: " + str(batch_words))
        np.array(model_2.wv.vectors).dump("node2vec_embedding_matrix.npy")
        self.logger.debug("Save file node2vec_embedding_matrix.npy")


    def run_all(self):
        self._custom_embedding()
        self._node2vec_embedding()



