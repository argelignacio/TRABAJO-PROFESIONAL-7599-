import tensorflow as tf
from tensorflow.keras.losses import CosineSimilarity

class TripletCustom(tf.keras.losses.Loss):
    def __init__(self, margin=0.2, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def __call__(self, y, y_pred, sample_weight=None):
        x_a, x_p, x_n = tf.split(y_pred, num_or_size_splits=3, axis=-1)
        gravity_const = 100

        distance_positive = 1 - CosineSimilarity()(x_a, x_p)
        distance_negative = 1 - CosineSimilarity()(x_a, x_n)

        loss = tf.maximum(gravity_const*(distance_positive) - (1/gravity_const)*distance_negative + self.margin, 0.0)
        return tf.reduce_mean(loss)
    