import numpy as np
import math

import tensorflow as tf


class PriorProbability(tf.keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        result = tf.ones(shape, dtype=dtype) * -tf.math.log((1 - self.probability) / self.probability)
        

        return tf.convert_to_tensor(result, dtype=dtype)