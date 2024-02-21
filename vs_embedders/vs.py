import os, sys
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import pickle as pkl

sys.path.insert(0, os.path.abspath(".."))
from clustering.embedders.processing_frames import pipeline_v2
from fake_graph.Fake_graph import GeneratorFakeGraph

df = GeneratorFakeGraph.generate_fake_graph_df(10, 500, 0.30, 500000)
df.to_csv("fake_graph.csv", index=False)

# First, run the custom embedding
embedding_matrix, ids = pipeline_v2(df)
np.array(embedding_matrix[0]).dump("embedding_matrix.npy")
pkl.dump(ids, open("ids.pkl", "wb"))

# Now run node2vec on the graph
G = nx.from_pandas_edgelist(df, 'from_address', 'to_address')
node2vec = Node2Vec(G, dimensions=128, walk_length=16, num_walks=100, workers=2)
model_2 = node2vec.fit(window=10, min_count=1, batch_words=4)
np.array(model_2.wv.vectors).dump("node2vec_embedding_matrix.npy")

# get the embeddings for the nodes
embedding_matrix_1 = np.load("embedding_matrix.npy", allow_pickle=True)
embedding_matrix_2 = np.load("node2vec_embedding_matrix.npy", allow_pickle=True)

print("Embedding matrix 1: ", embedding_matrix_1.shape)
print("Embedding matrix 2: ", embedding_matrix_2.shape)





