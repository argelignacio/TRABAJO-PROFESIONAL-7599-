from keras.utils import Sequence
import numpy as np
import pandas as pd
import tensorflow as tf

class GeneratorTriplet(Sequence):
    def __init__(self, df, ids, batch_size):
        self.df = df
        self.act_index = 0
        self.ids = ids
        self.batch_size = batch_size
        self.limit = int(np.ceil(len(self.df) / self.batch_size))

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    
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
        for i in range(self.batch_size):
              aux = self.df.sample(1)
              while (aux['to_address'] != batch.iloc[i]['to_address']) & (aux['to_address'] != batch.iloc[i]['from_address']):
                  aux = self.df.sample(1)
                  negative.append(aux)
        # negative = self.df.sample(len(batch)

        anchor = np.array(batch['from_address'].apply(lambda x: self.ids.get(x)))
        positive = np.array(batch['to_address'].apply(lambda x: self.ids.get(x)))
        negative = np.array(negative['to_address'].apply(lambda x: self.ids.get(x)))
        
        anchor = tf.convert_to_tensor(anchor)
        positive = tf.convert_to_tensor(positive)
        negative = tf.convert_to_tensor(negative)
        
        # el fake target simunla lo que sería aprendizaje supervisado        
        fake_target = tf.convert_to_tensor(np.array([1]*self.batch_size))

        return ([anchor, positive, negative], [fake_target]*3)