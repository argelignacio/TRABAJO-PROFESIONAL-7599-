import pandas as pd
import random
import uuid
import networkx as nx
from logger.logger import MyLogger

logger = MyLogger(__name__)

class GeneratorFakeGraph:
    def __init__(self) -> None:
        pass

    def generate_fake_graph_df(n_clusters, n_nodes_per_cluster, probability_intern=0.65, n_transactions=300):
        logger.info("Generating Fake Graph with: n_clusters: " + str(n_clusters) + " n_nodes_per_cluster: " + str(n_nodes_per_cluster) + " probability_intern: " + str(probability_intern) + " n_transactions: " + str(n_transactions))
        clusters = [[(str(uuid.uuid4()), cluster_id) for _ in range(n_nodes_per_cluster)] for cluster_id in range(n_clusters)]

        data = []
        for i in range(n_transactions):
            if random.random() < probability_intern:
                cluster_aux = int(random.random() * (n_clusters))
                id_1 = clusters[cluster_aux][int(random.random() * (n_nodes_per_cluster))]
                id_2 = clusters[cluster_aux][int(random.random() * (n_nodes_per_cluster))]
                new_row = {
                'from_address': id_1[0], 
                'to_address': id_2[0], 
                'cluster_from': id_1[1], 
                'cluster_to': id_2[1], 
                'block_timestamp': 1, 
                'value': (random.random()*1000)
                }
            else:
                id_1 = clusters[int(random.random() * (n_clusters))][int(random.random() * (n_nodes_per_cluster))]
                id_2 = clusters[int(random.random() * (n_clusters))][int(random.random() * (n_nodes_per_cluster))]
                new_row = {
                'from_address': id_1[0], 
                'to_address': id_2[0], 
                'cluster_from': id_1[1], 
                'cluster_to': id_2[1], 
                'block_timestamp': 1, 
                'value': (random.random()*200)
                }
            

            data.append(new_row)
        logger.info("Fake Graph was built")
        return pd.DataFrame(columns=['from_address', 'to_address', 'cluster_from', 'cluster_to', 'block_timestamp', 'value'], data=data)
    
    def generate_fake_graph(n_clusters, n_nodes_per_cluster, probability_intern=0.65, n_transactions=150):
        df = GeneratorFakeGraph.generate_fake_graph_df(n_clusters, n_nodes_per_cluster, probability_intern, n_transactions)
        G = nx.from_pandas_edgelist(df, 'from_address', 'to_address')
        return G
    
