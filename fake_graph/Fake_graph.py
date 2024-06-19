import pandas as pd
import random
import uuid
import networkx as nx
import numpy as np

class GeneratorFakeGraph:
    def __init__(self) -> None:
        pass

    
    def generate_fake_graph(logger, n_clusters, n_nodes_per_cluster, probability_intern=0.65, n_transactions=150):
        df = GeneratorFakeGraph.generate_fake_graph_df(logger, n_clusters, n_nodes_per_cluster, probability_intern, n_transactions)
        G = nx.from_pandas_edgelist(df, 'from_address', 'to_address')
        return G

    
    def generate_fake_graph_df(logger, n_clusters, n_nodes_per_cluster, probability_intern=0.65, n_transactions=300, floating_nodes_proportion=0.3, intern_ratio=1, n_nodes_per_cluster_deviation=0):
            logger.info(f"Generating Fake Graph with: n_clusters: {str(n_clusters)} n_nodes_per_cluster: {str(n_nodes_per_cluster)} probability_intern: {str(probability_intern)} n_transactions: {str(n_transactions)} floating_nodes_proportion: {str(floating_nodes_proportion)}")
            clusters = []
            nodes_per_cluster = list(np.random.normal(loc=n_nodes_per_cluster, scale=n_nodes_per_cluster_deviation, size=n_clusters))
            for cluster_id in range(n_clusters):
                cluster = [(str(uuid.uuid4()), cluster_id) for _ in range(int(nodes_per_cluster[cluster_id]))]
                clusters.append(cluster)
            
            floating_count = int(n_clusters * n_nodes_per_cluster * floating_nodes_proportion)
            if floating_nodes_proportion > 0:  
                noise_nodes =  [[(str(uuid.uuid4()), -1) for _ in range(floating_count)]]
                clusters = clusters + noise_nodes
                nodes_per_cluster.append(floating_count)
            
            data = []
            timestamp = 0
            for _ in range(n_transactions): 
                if random.random() < (floating_count/(n_clusters * n_nodes_per_cluster+floating_count)):
                    cluster_aux = random.randint(0, n_clusters)
                    if random.random() < 0.5:
                        id_1 = clusters[cluster_aux][int(random.random() * (int(nodes_per_cluster[cluster_aux])))]
                        id_2 = clusters[-1][int(random.random() * (floating_count))]
                    else:
                        id_1 = clusters[-1][int(random.random() * (floating_count))]
                        id_2 = clusters[cluster_aux][int(random.random() * (int(nodes_per_cluster[cluster_aux])))]
                    new_row = {
                        'from_address': id_1[0], 
                        'to_address': id_2[0], 
                        'cluster_from': id_1[1], 
                        'cluster_to': id_2[1], 
                        'block_timestamp': timestamp, 
                        'value': (random.random() * 1000*  intern_ratio)
                    }          
                elif random.random() < probability_intern:
                    cluster_aux = int(random.random() * (n_clusters))
                    id_1 = clusters[cluster_aux][int(random.random() * (int(nodes_per_cluster[cluster_aux])))]
                    id_2 = clusters[cluster_aux][int(random.random() * (int(nodes_per_cluster[cluster_aux])))]
                    new_row = {
                        'from_address': id_1[0], 
                        'to_address': id_2[0], 
                        'cluster_from': id_1[1], 
                        'cluster_to': id_2[1], 
                        'block_timestamp': timestamp, 
                        'value': (random.random() * 1000)
                    }
                else:
                    cluster_aux_1 = int(random.random() * (n_clusters))
                    id_1 = clusters[cluster_aux_1][int(random.random() * (int(nodes_per_cluster[cluster_aux_1])))]
                    cluster_aux_2 = int(random.random() * (n_clusters))
                    id_2 = clusters[cluster_aux_2][int(random.random() * (int(nodes_per_cluster[cluster_aux_2])))]
                    new_row = {
                        'from_address': id_1[0], 
                        'to_address': id_2[0], 
                        'cluster_from': id_1[1], 
                        'cluster_to': id_2[1], 
                        'block_timestamp': timestamp, 
                        'value': (random.random() * 1000 * intern_ratio)
                    }
                data.append(new_row)
                timestamp += 1
            logger.debug("Fake Graph was built")
            return pd.DataFrame(columns=['from_address', 'to_address', 'cluster_from', 'cluster_to', 'block_timestamp', 'value'], data=data)