import numpy as np
import tensorflow as tf

class DistanceLayer(tf.keras.layers.Layer):
    def __init__(self, metric='euclidean', dynamic=False):
        super().__init__(dynamic=dynamic)
        self.metric = metric
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0],)
    
    def call(self, s1, s2):
        if self.metric == 'euclidean':
            d = tf.reduce_sum(tf.square(s1 - s2), -1)
            return d 
        elif self.metric == 'hyperbolic':
            #print(s1.numpy())
            #HYP_EPSILON = 1e-6
            sqdist = tf.reduce_sum((s1 - s2) ** 2, axis = -1)
            squnorm = tf.reduce_sum(s2 ** 2 , axis = -1)
            sqvnorm = tf.reduce_sum(s1 ** 2 , axis = -1)
            #squnorm = tf.clip_by_value( squnorm, 0, 1-HYP_EPSILON )
            #sqvnorm = tf.clip_by_value( sqvnorm, 0, 1-HYP_EPSILON )
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

class OneHotEncodingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.vocab = 4
    
    def call(self, x):
        return tf.one_hot(tf.cast(x, tf.int32), self.vocab)
        
        
        