import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import pandas as pd
from clustering.embedders.all_v2.GeneratorV2 import GeneratorTriplet
from clustering.embedders.all_v1.Loss import EuclideanLoss
from clustering.embedders.all_v2.ModelV2 import ModelBuilder
import os
from datetime import datetime, timedelta

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
        df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
        df['value'] = df['value'].astype('float64') / 1e18
        processing_frames = ProcessingFrames(df, logger)
        return processing_frames
    
    def build_from_df(df, logger):
        processing_frames = ProcessingFrames(df, logger)
        return processing_frames

    def _clean_nodes(self):
        df_tmp = self.df
        df_tmp = df_tmp[~df_tmp.to_address.isna()]
        df_tmp = df_tmp[~df_tmp.from_address.isna()]
        df_tmp = df_tmp[df_tmp["from_address"] != df_tmp["to_address"]]
        uniques = df_tmp['from_address']._append(df_tmp['to_address']).value_counts()
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

    def pipeline(self, config):
        cleaned_df = self._clean_nodes()
        addresses_ids = self._create_ids(cleaned_df)

        self.logger.debug("Creating model")
        model_v2_config = config["MODEL_V2"]
        model = ModelBuilder(addresses_ids, EuclideanLoss, Adam, self.logger, model_v2_config)
        
        generator = self._create_generator(cleaned_df, addresses_ids, config)
        embeddings = model.compile_model().fit(generator).get_embeddings()
        return embeddings, addresses_ids

def main(logger):
    processing_frames = ProcessingFrames()
    processing_frames.set_gpu(logger)
    months = ["July"]
    src = "../../../datos"
    generated_files = processing_frames.generate_dates(src, months)
    for files in generated_files:
        processing_frames.pipeline(files)