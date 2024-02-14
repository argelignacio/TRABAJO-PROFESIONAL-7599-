import pandas as pd
import networkx as nx 
from node2vec import Node2Vec

df = pd.concat(map(pd.read_csv, ['../../datos/july_23/2023-07-01.csv', '../../datos/july_23/2023-07-02.csv', '../../datos/july_23/2023-07-03.csv']))

G = nx.from_pandas_edgelist(df, source='from_address', target='to_address', create_using=nx.DiGraph())
try:
    print("Inicia Node2Vec")
    node2vec = Node2Vec(G, dimensions=64, walk_length=80, num_walks=10, workers=2)

    print("Inicia node2vec.fit")
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    print("guardando resultado")
    model.wv.save_word2vec_format('node2vec.emb')
    print("done")
except e:
    print(e)