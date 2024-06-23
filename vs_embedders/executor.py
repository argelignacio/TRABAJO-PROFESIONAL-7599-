import os, sys
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import pickle as pkl

sys.path.insert(0, os.path.abspath(".."))
from clustering.embedders.processing_frames import ProcessingFrames

class Executor:
    def __init__(self, logger, df, config, file_management) -> None:
        self.logger = logger
        self.df = df
        self.config = config
        self.file_management = file_management

    def _calculate_embedding_sizes(self):
        model_config = self.config["MODEL_V2"]
        current_embedding_size = int(model_config["embedding_dim"])
        embedding_sizes = []
        while current_embedding_size >= 2:
            embedding_sizes.append(current_embedding_size)
            current_embedding_size = current_embedding_size // 2
        return embedding_sizes


    def _custom_embedding(self):
        processing_frames = ProcessingFrames.build_from_df(self.df, self.logger)
        for embedding_size in self._calculate_embedding_sizes():
            config = pkl.loads(pkl.dumps(self.config))
            config["MODEL_V2"]["embedding_dim"] = str(embedding_size)
            embedding_matrix, ids = processing_frames.pipeline(config)
            self.file_management.save_npy(f"embedding_matrix_{embedding_size}.npy", embedding_matrix[0])
            self.logger.debug(f"Saved file embedding_matrix_{embedding_size}.npy")
        self.file_management.save_pkl("ids.pkl", ids)
        self.logger.debug(f"Saved file ids.pkl")

    def _node2vec_embedding(self, dimensions=128, walk_length=200, num_walks=100, workers=2, window=10, min_count=1, batch_words=4):
        G = nx.from_pandas_edgelist(self.df, 'from_address', 'to_address')
        node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
        self.logger.info("Node2Vec model created with dimensions: " + str(dimensions) + " walk_length: " + str(walk_length) + " num_walks: " + str(num_walks) + " workers: " + str(workers))
        
        model_2 = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
        self.logger.info("Node2Vec model fitted with window: " + str(window) + " min_count: " + str(min_count) + " batch_words: " + str(batch_words))
        
        self.file_management.save_npy("node2vec_embedding_matrix.npy", np.array(model_2.wv.vectors))
        self.logger.debug(f"Saved file node2vec_embedding_matrix.npy")

    def run_all(self):
        self._custom_embedding()
        self._node2vec_embedding()



