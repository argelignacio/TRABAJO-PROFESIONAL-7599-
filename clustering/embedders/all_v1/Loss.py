import tensorflow as tf

class EuclideanLoss(tf.keras.losses.Loss):
    def __init__(self, margin=0.2, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def __call__(self, y, y_pred, sample_weight=None):
        anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=-1)

        x_a = anchor[:,:-1]
        x_p = positive[:,:-1]
        x_n = negative[:,:-1]
        gravity_const = anchor[:,-1]

        distance_positive = tf.reduce_sum(tf.square(x_a - x_p), axis=1)
        distance_negative = tf.reduce_sum(tf.square(x_a - x_n), axis=1)

        distance_positive = gravity_const * distance_positive

        loss = tf.maximum(distance_positive - distance_negative + self.margin, 0.0)
        return tf.reduce_mean(loss)
    
    def name():
        return 'EuclidianDistanceLoss'