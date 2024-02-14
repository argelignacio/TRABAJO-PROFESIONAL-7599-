import pandas as pd
import random
import uuid
import networkx as nx

class GeneratorFakeGraph:
    def __init__(self) -> None:
        pass

    def generate_fake_graph_df(n_clusters, n_nodes_per_cluster, probability_intern=0.65, n_transactions=150):
        clusters = [[uuid.uuid4() for _ in range(n_nodes_per_cluster)] for _ in range(n_clusters)]
        
        data = []
        for _ in range(n_transactions):
            if random.random() < probability_intern:
                cluster_aux = int(random.random() * (n_clusters -1))
                id_1 = clusters[cluster_aux][int(random.random() * (n_nodes_per_cluster - 1))]
                id_2 = clusters[cluster_aux][int(random.random() * (n_nodes_per_cluster - 1))]
            else:
                id_1 = clusters[int(random.random() * (n_clusters -1))][int(random.random() * (n_nodes_per_cluster - 1))]
                id_2 = clusters[int(random.random() * (n_clusters -1))][int(random.random() * (n_nodes_per_cluster - 1))]
                
            new_row = {'from_address': id_1, 'to_address': id_2}
            data.append(new_row)
        return pd.DataFrame(columns=['from_address', 'to_address'], data=data)
    
    def generate_fake_graph(n_clusters, n_nodes_per_cluster, probability_intern=0.65, n_transactions=150):
        df = GeneratorFakeGraph.generate_fake_graph_df(n_clusters, n_nodes_per_cluster, probability_intern, n_transactions)
        G = nx.from_pandas_edgelist(df, 'from_address', 'to_address')
        return G
    
