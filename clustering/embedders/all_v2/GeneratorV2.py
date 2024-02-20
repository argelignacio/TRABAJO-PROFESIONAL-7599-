from keras.utils import Sequence
import numpy as np
import pandas as pd
import tensorflow as tf

class GeneratorTriplet(Sequence):
    def __init__(self, df, ids, batch_size):
        self.df = self.reduce_df(df)
        self.act_index = 0
        self.ids = ids
        self.batch_size = batch_size
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
        df['gravity_const'] = df['total_amount']/df['count_transactions'] 
        df['gravity_const'] =6*((df['gravity_const'] - df['gravity_const'].mean() ) / df['gravity_const'].std())
        return df
    
    def init_positives(self):
        positives = {}
        for from_add in self.df['from_address']:
            positives[from_add] = self.df[self.df.from_address == from_add]['to_address'].values
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

        return ([anchor, metadata, positive, metadata, negative, metadata], [fake_target]*6)