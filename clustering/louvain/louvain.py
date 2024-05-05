import networkx as nx
import pandas as pd

class Louvain():
    def __init__(self, logger, config, file_management):
        self.logger = logger
        self.config = config
        self.resolution = float(config["LOUVAIN"]["resolution"])
        self.weight = config["LOUVAIN"]["weight"]
        self.threshold = float(config["LOUVAIN"]["threshold"])
        self.file_management = file_management

    def run_louvain(self, df):
        self.logger.info("Running Louvain")
        G = nx.from_pandas_edgelist(df, 'from_address', 'to_address', 'value')
        partition = nx.community.louvain_communities(G, resolution=self.resolution, weight=self.weight, threshold=self.threshold)
        return partition

    def mapping_partition_to_df(self, partition):
        node_community_mapping = {}
        for community_id, nodes in enumerate(partition):
            for node in nodes:
                node_community_mapping[node] = community_id

        df_labels_louvain = pd.DataFrame.from_dict(node_community_mapping, orient='index', columns=['louvain']).reset_index()
        df_labels_louvain.rename(columns={'index': 'node_ids'}, inplace=True)
        return df_labels_louvain


    def run(self, df):
        partition = self.run_louvain(df)
        df_labels = self.mapping_partition_to_df(partition)
        self.file_management.save_df("louvain_community.csv", df_labels)
        return df_labels
