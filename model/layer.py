import numpy as np
import tensorflow as tf

class DistanceLayer(tf.keras.layers.Layer):
    def __init__(self, metric='euclidean'):
        super().__init__()
        self.metric = metric
    
    def call(self, s1, s2):
        if self.metric == 'euclidean':
            d = tf.reduce_sum(tf.square(s1 - s2), -1)
            return d 
        elif self.metric == 'hyperbolic':
            # TODO: implement hyperbolic distance: https://github.com/gcorso/NeuroSEED/blob/master/util/distance_functions/distance_functions.py
            sqdist = tf.reduce_sum((s1 - s2) ** 2, axis = -1)
            squnorm = tf.reduce_sum(s2 ** 2 , axis = -1)
            sqvnorm = tf.reduce_sum(s1 ** 2 , axis = -1)
            x = 1 + ( 2 * (sqdist / ((1 - squnorm)*(1 - sqvnorm)) ))
            z = tf.math.sqrt( (x**2) - 1)
            d = tf.math.log(x + z)
            return d
        elif self.metric == 'manhattan':
            d = tf.reduce_sum( tf.math.abs( s1 - s2), -1)
            return d
        elif self.metric == 'square':
            D = s1 - s2
            d = tf.reduce_sum(D * D, axis=-1)
            return d
        elif self.metric == 'cosine':
            return 1 - tf.keras.losses.cosine_similarity(s1,s2)
        else:
            raise ValueError(f"No metric named '{self.metric}' exists")