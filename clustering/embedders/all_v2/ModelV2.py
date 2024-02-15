from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, GaussianNoise

class Model():
    def __init__(self):
        pass
    
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

        x_a = self.model_aux(input_layer_anchor)
        x_p = self.model_aux(input_layer_positive)
        x_n = self.model_aux(input_layer_negative)

        merged_output = Concatenate(axis=-1)([x_a, x_p, x_n])

        self.model = Model([input_layer_anchor, input_layer_positive, input_layer_negative], merged_output)
        return self.model