import tensorflow as tf
from tensorflow.keras.losses import CosineSimilarity

class TripletCustom(tf.keras.losses.Loss):
    def __init__(self, margin=0.2, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def __call__(self, y, y_pred, sample_weight=None):
        anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=-1)
        x_a = tf.slice(anchor, [0, 0], [0, 63])
        x_p = tf.slice(positive, [0, 0], [0, 63])
        x_n = tf.slice(negative, [0, 0], [0, 63])
        metadata = tf.slice(anchor, [0, 63], [0, 2])

        # trxs_count, value_sum
        gravity_const, _ = tf.split(metadata, num_or_size_splits=2, axis=-1)

        distance_positive = 1 - CosineSimilarity()(x_a, x_p)
        distance_negative = 1 - CosineSimilarity()(x_a, x_n)

        print(distance_positive)
        gravity_const = tf.squeeze(gravity_const)
        print(gravity_const)

        distance_positive = tf.math.scalar_mul(gravity_const, distance_positive)
        distance_negative = tf.math.scalar_mul(1/gravity_const, distance_negative)

        loss = tf.maximum(distance_positive - distance_negative + self.margin, 0.0)
        return tf.reduce_mean(loss)
    