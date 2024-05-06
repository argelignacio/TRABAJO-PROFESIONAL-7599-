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
        
        df_temp = df.groupby('from_address').agg({'count_transactions': 'sum', 'total_amount': 'sum' })

        df['transaction_importance_pair'] = df['count_transactions'] / df['from_address'].map(df_temp['count_transactions'])
        df['amount_importance_pair'] = df['total_amount'] / df['from_address'].map(df_temp['total_amount'])

        weigth = float(self.config["GENERATOR_V2"]["weigth"])
        
        df['gravity_const'] = weigth*(df['transaction_importance_pair'] + df['amount_importance_pair'])        
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