from keras.utils import Sequence
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import sys
import os
sys.path.insert(0, os.path.abspath("../../.."))

class GeneratorTriplet(Sequence):
    def __init__(self, df, ids, config, logger):
        self.logger = logger
        self.config = config
        self.df = self.reduce_df(df)
        self.act_index = 0
        self.ids = ids
        self.batch_size = int(config["GENERATOR_V2"]["batch_size"])
        self.limit = int(np.ceil(len(self.df) / self.batch_size))
        self.positives = self.init_positives()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def reduce_df(self, df):
        df = df\
            .groupby(['from_address', 'to_address'])\
            .agg({ 'block_timestamp': 'count', 'value': 'sum' })\
            .rename(columns={'block_timestamp': 'count_transactions', 'value': 'total_amount'})\
            .reset_index()
        
        weigth = int(self.config["GENERATOR_V2"]["weigth"])

        # lo que intenta es darle un peso a aquellos que frecuentan transacciones
        normalized_count_transactions = (df['count_transactions'] - df['count_transactions'].mean()) / df['count_transactions'].std()

        # lo que intenta es darle un peso a aquellos que envian mas dinero
        average_amount = df['total_amount'] / df['count_transactions']
        gravity_const_mean = average_amount.mean()
        gravity_const_std = average_amount.std()
        normalized_gravity_const = (average_amount - gravity_const_mean) / gravity_const_std

        # TODO: podriamos tener dos weigth diferentes teniendo en cuenta que podria tener diferentes importancias las frecuencias y los montos
        # TODO: buscar una relacion para los weigth y que no sea estático
        df['gravity_const'] = weigth*(normalized_gravity_const + normalized_count_transactions)
        self.logger.debug("Dataframe reduced.")
        return df

    def init_positives(self):
        positives = self.df.groupby('from_address')['to_address'].apply(list).to_dict()
        self.logger.debug("Positives initialized.")
        return positives
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.act_index < self.limit:
            resultado = self.__getitem__(self.act_index)
            self.act_index += 1
            return resultado
        else:
            raise StopIteration

    def __getitem__(self, index):
        # el Anchor es un A, el positive es un B y el negative es un C. Queremos acercar A y B tanto como sea posible alejando a C. Podría suceder que sea un vecino, pero la probabilidad es baja porque es al azar.
        # Si llegara a elegir algún vecino mal, de casualidad, en una siguiente epoch debería corregirse o inclusive en una misma epoch en una siguiente iteración del generador.
        init = index * self.batch_size
        end = (index + 1) * self.batch_size

        # el batch van a ser #batch_size tuplas que representan el anchor y positive. El negative es un random.
        batch = self.df[init:end]

        # me agarro un sample de #batch_size transacciones.
        negative = []
        for i in range(len(batch)):
            aux = self.df.sample(1)
            while aux['to_address'].values[0] in self.positives[batch.iloc[i]['from_address']]:
                aux = self.df.sample(1)
            negative.append(aux['to_address'].values[0])

        metadata = np.array(batch['gravity_const'].values)
        
        anchor = np.array(batch['from_address'].apply(lambda x: self.ids.get(x)))
        positive = np.array(batch['to_address'].apply(lambda x: self.ids.get(x)))
        negative = np.array(list(map(lambda x: self.ids.get(str(x)), negative)))

        anchor = tf.convert_to_tensor(anchor)
        positive = tf.convert_to_tensor(positive)
        negative = tf.convert_to_tensor(negative)
        metadata = tf.convert_to_tensor(metadata)

        # el fake target simunla lo que sería aprendizaje supervisado        
        fake_target = tf.convert_to_tensor(np.array([1]*self.batch_size))

        true_target = [anchor, metadata, positive, metadata, negative, metadata]
        return (true_target, [fake_target] * len(true_target))