import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, GaussianNoise, Reshape

class ModelBuilder():
    def __init__(self, ids, loss, optimizer):
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
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
        self.model.fit(generator, epochs=1000, callbacks=[callback])
        self.trained = True
        return self

    def compile_model(self, lr=2e-4):
        self.model.compile(
            optimizer=self.optimizer(lr),
            loss=self.loss()
        )
        return self

    def create_embedder(self, ids):
        input_aux = Input(1)
        x = Embedding(len(ids), 128)(input_aux)
        x = GaussianNoise(0.02)(x)
        x = Dense(64)(x)
        output_aux = Dense(64)(x)
        self.model_aux = Model(input_aux, output_aux)
        return self.model_aux

    def print_embedder(self):
        self.model_aux.summary()

    def print_wrapper(self):
        self.model.summary()

    def create_wrapper(self):
        input_layer_anchor = Input(1)
        input_layer_positive = Input(1)
        input_layer_negative = Input(1)
        metadata = Input(1)
        metadata2 = Input(1)
        metadata3 = Input(1)

        x_a = Reshape((64,))(self.model_aux(input_layer_anchor))
        x_p = Reshape((64,))(self.model_aux(input_layer_positive))
        x_n = Reshape((64,))(self.model_aux(input_layer_negative))

        merged_a = Concatenate()([x_a, metadata])
        merged_p = Concatenate()([x_p, metadata2])
        merged_n = Concatenate()([x_n, metadata3])
        merged_output = Concatenate(axis=-1)([merged_a, merged_p, merged_n])
        
        self.model = Model([input_layer_anchor, metadata, input_layer_positive, metadata2, input_layer_negative, metadata3], merged_output)
        return self.model
