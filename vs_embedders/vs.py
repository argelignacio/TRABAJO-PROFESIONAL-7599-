import os, sys
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import pickle as pkl

sys.path.insert(0, os.path.abspath(".."))
from clustering.embedders.processing_frames import pipeline_v2
from fake_graph.Fake_graph import GeneratorFakeGraph
from logger.logger import MyLogger

logger = MyLogger(__name__)

def custom_embedding(df):
    embedding_matrix, ids = pipeline_v2(df)
    np.array(embedding_matrix[0]).dump("custom_embedding_matrix.npy")
    logger.info("Save file custom_embedding_matrix.npy")
    pkl.dump(ids, open("ids.pkl", "wb"))
    logger.info("Save file ids.pkl")

def node2vec_embedding(df, dimensions=128, walk_length=16, num_walks=100, workers=2, window=10, min_count=1, batch_words=4):
    G = nx.from_pandas_edgelist(df, 'from_address', 'to_address')
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
    logger.info("Node2Vec model created with dimensions: " + str(dimensions) + " walk_length: " + str(walk_length) + " num_walks: " + str(num_walks) + " workers: " + str(workers))
    model_2 = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
    logger.info("Node2Vec model fitted with window: " + str(window) + " min_count: " + str(min_count) + " batch_words: " + str(batch_words))
    np.array(model_2.wv.vectors).dump("node2vec_embedding_matrix.npy")
    logger.info("Save file node2vec_embedding_matrix.npy")


def run_all(df):
    custom_embedding(df)
    node2vec_embedding(df)



