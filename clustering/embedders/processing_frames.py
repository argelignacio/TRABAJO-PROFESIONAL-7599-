import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import pandas as pd
from clustering.embedders.all_v2.GeneratorV2 import GeneratorTriplet
from clustering.embedders.all_v1.Loss import EuclideanLoss
from clustering.embedders.all_v2.ModelV2 import ModelBuilder

class ProcessingFrames:
    def __init__(self, df, logger) -> None:
        self.logger = logger
        self.df = df
        ProcessingFrames.set_gpu(logger)

    def set_gpu(logger):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.set_visible_devices(gpus[0], 'GPU')
                logical_gpus = tf.config.list_logical_devices('GPU')
                logger.debug(f"GPU set: {logical_gpus}")
            except RuntimeError as e:
                logger.error(e)

    def build_from_files(files, logger):
        df = pd.concat((pd.read_csv(f, nrows=70000) for f in files), ignore_index=True)
        df['value'] = df['value'].astype('float64') / 1e18
        processing_frames = ProcessingFrames(df, logger)
        return processing_frames
    
    def build_from_df(df, logger):
        processing_frames = ProcessingFrames(df, logger)
        return processing_frames

    def _clean_nodes(self):
        self.logger.info("Cleaning nodes")
        df_tmp = self.df
        df_tmp = df_tmp[~df_tmp.to_address.isna()]
        df_tmp = df_tmp[~df_tmp.from_address.isna()]
        df_tmp = df_tmp[df_tmp["from_address"] != df_tmp["to_address"]]
        uniques = df_tmp['from_address'].append(df_tmp['to_address']).value_counts()
        unique_values = uniques[uniques == 1]
        return df_tmp[~(df_tmp['from_address'].isin(unique_values.index) | df_tmp['to_address'].isin(unique_values.index))]

    def _create_ids(self, cleaned_df):
        ids = {}
        for i, id in enumerate(set(cleaned_df['from_address']).union(set(cleaned_df['to_address']))):
            ids[id] = i
        self.logger.debug("Ids created")
        return ids

    def _create_generator(self, cleaned_df, addresses_ids, config):
        generator = GeneratorTriplet(cleaned_df, addresses_ids, config, self.logger)
        self.logger.debug("Generator created")
        return generator

    def pipeline(self, config, pending_model):
        cleaned_df = self._clean_nodes()
        min_transactions = self._search_max_transaccions_below_percentage(cleaned_df)
        filtered_df = self._filter_nodes_per_min_transactions(cleaned_df, min_transactions + 1)
        addresses_ids = self._create_ids(filtered_df)


        self.logger.debug("Creating model")
        model_v2_config = config["MODEL_V2"]
        model = ModelBuilder(addresses_ids, EuclideanLoss, Adam, self.logger, model_v2_config)
        
        generator = self._create_generator(filtered_df, addresses_ids, config)
        embeddings = model.compile_model().fit(generator, pending_model).get_embeddings()
        return embeddings, addresses_ids
    
    def _filter_per_transactions(self, df, percentage=0.1):
        self.logger.debug("Filtering transactions")
        total_transactions = len(df)
        self.logger.info(f"Total transactions: {total_transactions}")

        all_addresses = df['from_address'].tolist() + df['to_address'].tolist()
        # Count the transactions per address
        address_counts = pd.Series(all_addresses).value_counts().reset_index()
        address_counts.columns = ['address', 'transaction_count']
        self.logger.info(f"Len address counts: {len(address_counts)}")
        self.logger.info(f"Firsts addresses with most transactions: {address_counts.head()}")

        # Calculate the threshold to remove the addresses
        threshold = int(len(address_counts) * percentage)

        # Get the addresses to remove
        addresses_to_remove = address_counts.nsmallest(threshold, 'transaction_count')['address']
        self.logger.info(f"Len addresses to remove: {len(addresses_to_remove)}")

        # Filter the transactions that have the addresses to remove
        filtered_df = df[(~df['from_address'].isin(addresses_to_remove)) & (~df['to_address'].isin(addresses_to_remove))]

        self.logger.info(f"Len filtered transactions: {len(filtered_df)}")
        return filtered_df

    def _filter_nodes_per_min_transactions(self, df, min_transactions=10):
        self.logger.info(f"Filtering transactions with at least {min_transactions} transactions")
        self.logger.info(f"Total transactions: {len(df)}")

        all_addresses = df['from_address'].tolist() + df['to_address'].tolist()
        # Count the transactions per address
        address_counts = pd.Series(all_addresses).value_counts().reset_index()
        address_counts.columns = ['address', 'transaction_count']
        self.logger.info(f"Len address counts: {len(address_counts)}")
        self.logger.info(f"Firsts addresses with most transactions: {address_counts.head()}")


        # Get the addresses to remove
        addresses_to_remove = address_counts[address_counts['transaction_count'] < min_transactions]['address']
        self.logger.info(f"Len addresses to remove: {len(addresses_to_remove)}")

        # Filter the transactions that have the addresses to remove
        filtered_df = df[(~df['from_address'].isin(addresses_to_remove)) & (~df['to_address'].isin(addresses_to_remove))]

        self.logger.info(f"Len filtered transactions: {len(filtered_df)}")
        return filtered_df

    def _search_max_transaccions_below_percentage(self, df, percentage=0.1):
        self.logger.info(f"Search max transactions below percentage: {percentage}")
        all_addresses = df['from_address'].tolist() + df['to_address'].tolist()
        # Count the transactions per address
        address_counts = pd.Series(all_addresses).value_counts().reset_index()
        address_counts.columns = ['address', 'transaction_count']

        threshold = int(len(address_counts) * percentage)

        # Get the max transactions below the threshold 
        max_transactions = address_counts.nsmallest(threshold, 'transaction_count')['transaction_count'].max()

        self.logger.info(f"Max transactions: {max_transactions}")
        return max_transactions

def main(logger):
    processing_frames = ProcessingFrames()
    processing_frames.set_gpu(logger)
    months = ["July"]
    src = "../../../datos"
    generated_files = processing_frames.generate_dates(src, months)
    for files in generated_files:
        processing_frames.pipeline(files)