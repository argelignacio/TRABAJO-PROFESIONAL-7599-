import pandas as pd
import random
import uuid
import networkx as nx

class GeneratorFakeGraph:
    def __init__(self) -> None:
        pass

    
    def generate_fake_graph(logger, n_clusters, n_nodes_per_cluster, probability_intern=0.65, n_transactions=150):
        df = GeneratorFakeGraph.generate_fake_graph_df(logger, n_clusters, n_nodes_per_cluster, probability_intern, n_transactions)
        G = nx.from_pandas_edgelist(df, 'from_address', 'to_address')
        return G

    
    def generate_fake_graph_df(logger, n_clusters, n_nodes_per_cluster, probability_intern=0.65, n_transactions=300, floating_nodes_proportion=0.3):
            logger.info(f"Generating Fake Graph with: n_clusters: {str(n_clusters)} n_nodes_per_cluster: {str(n_nodes_per_cluster)} probability_intern: {str(probability_intern)} n_transactions: {str(n_transactions)}")
            clusters = [[(str(uuid.uuid4()), cluster_id) for _ in range(n_nodes_per_cluster)] for cluster_id in range(n_clusters)]

            if floating_nodes_proportion > 0:
                floating_count = int(n_clusters * n_nodes_per_cluster * floating_nodes_proportion)
                noise_nodes =  [[(str(uuid.uuid4()), -1) for _ in range(floating_count)]]
                clusters = clusters + noise_nodes
            
            data = []
            for _ in range(n_transactions):
                if random.random() < floating_nodes_proportion:
                    cluster_aux = int(random.random() * (n_clusters+1)) - 1
                    if random.random() < 0.5:
                        id_1 = clusters[cluster_aux][int(random.random() * (n_nodes_per_cluster))]
                        id_2 = clusters[-1][int(random.random() * (n_nodes_per_cluster))]
                    else:
                        id_1 = clusters[-1][int(random.random() * (n_nodes_per_cluster))]
                        id_2 = clusters[cluster_aux][int(random.random() * (n_nodes_per_cluster))]
                    new_row = {
                    'from_address': id_1[0], 
                    'to_address': id_2[0], 
                    'cluster_from': id_1[1], 
                    'cluster_to': id_2[1], 
                    'block_timestamp': 1, 
                    'value': (random.random()*1000)
                    }
                                        
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
                    'value': (random.random()*1000)
                    }
                

                data.append(new_row)
            logger.debug("Fake Graph was built")
            return pd.DataFrame(columns=['from_address', 'to_address', 'cluster_from', 'cluster_to', 'block_timestamp', 'value'], data=data)