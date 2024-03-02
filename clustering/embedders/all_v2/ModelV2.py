import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, GaussianNoise, Reshape
import time

class ModelBuilder():
    def __init__(self, ids, loss, optimizer, logger, model_config):
        self.model_config = model_config
        self.embedding_dim = int(self.model_config["embedding_dim"])
        self.logger = logger
        logger.info(f'Creating model with {loss.name()} loss and the embedding of {self.embedding_dim} dim.')
        self.loss = loss
        self.optimizer = optimizer
        self.embedder = self.create_embedder(ids)
        self.wrapper = self.create_wrapper()
        self.trained = False

    def get_embeddings(self):
        if self.trained:
            return self.model_aux.get_layer(name="embedding").get_weights()
        raise Exception("Modelo no entrenado, emb basura.")
    
    def fit(self, generator):
        self.logger.info('Train starting')
        conf = self.model_config
        start_fit = time.time()
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=int(conf["patience"]), restore_best_weights=True)
        self.model.fit(generator, epochs=int(conf["epochs"]), callbacks=[callback])
        end_fit = time.time()
        self.logger.info(f'Model fit in: {(end_fit - start_fit)/60} minutes.')
        self.trained = True
        return self

    def compile_model(self):
        self.model.compile(
            optimizer=self.optimizer(float(self.model_config["learning_rate"])),
            loss=self.loss()
        )
        return self

    def create_embedder(self, ids):
        input_aux = Input(1)
        x = Embedding(len(ids), self.embedding_dim)(input_aux)
        output_aux = GaussianNoise(float(self.model_config["gaussian_noise"]))(x)
        self.model_aux = Model(input_aux, output_aux)
        return self.model_aux

    def print_embedder(self):
        self.model_aux.summary()

    def print_wrapper(self):
        self.model.summary()

    def create_wrapper(self):
        conf = self.model_config

        input_layer_anchor = Input(1)
        input_layer_positive = Input(1)
        input_layer_negative = Input(1)
        metadata = Input(1)
        metadata2 = Input(1)
        metadata3 = Input(1)

        embedding_dim = int(conf["embedding_dim"])
        x_a = Reshape((embedding_dim,))(self.model_aux(input_layer_anchor))
        x_p = Reshape((embedding_dim,))(self.model_aux(input_layer_positive))
        x_n = Reshape((embedding_dim,))(self.model_aux(input_layer_negative))

        merged_a = Concatenate()([x_a, metadata])
        merged_p = Concatenate()([x_p, metadata2])
        merged_n = Concatenate()([x_n, metadata3])
        merged_output = Concatenate(axis=-1)([merged_a, merged_p, merged_n])
        
        self.model = Model([input_layer_anchor, metadata, input_layer_positive, metadata2, input_layer_negative, metadata3], merged_output)
        return self.model
