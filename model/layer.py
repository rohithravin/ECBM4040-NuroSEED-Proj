import numpy as np
import tensorflow as tf

class DistanceLayer(tf.keras.layers.Layer):
    def __init__(self, metric='euclidean', dynamic=False):
        super().__init__(dynamic=dynamic)
        self.metric = metric
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0],)
    
    # Decorate with tf.function to fix model saving
    @tf.function
    def call(self, s1, s2):
        if self.metric == 'euclidean':
            d = tf.reduce_sum(tf.square(s1 - s2), -1)
            return d 
        elif self.metric == 'hyperbolic':
            #print(s1.numpy())
            HYP_EPSILON = 1e-6
            sqdist = tf.reduce_sum((s1 - s2) ** 2, axis = -1)
            squnorm = tf.reduce_sum(s2 ** 2 , axis = -1)
            sqvnorm = tf.reduce_sum(s1 ** 2 , axis = -1)
            #squnorm = tf.clip_by_value( squnorm, 0, 1-HYP_EPSILON )
            #sqvnorm = tf.clip_by_value( sqvnorm, 0, 1-HYP_EPSILON )
            divisor = tf.math.maximum( 1 - squnorm, tf.constant(HYP_EPSILON) ) * tf.math.maximum( 1 - sqvnorm, tf.constant(HYP_EPSILON) )
            #divisor = tf.math.maximum( (1 - squnorm) * (1 - sqvnorm), tf.constant(HYP_EPSILON) )
            divisor = tf.math.maximum( divisor, tf.constant(HYP_EPSILON) )
            x = 1 + 2 * sqdist / divisor
            z = tf.math.sqrt( x**2 - 1) if not tf.reduce_any(tf.math.is_inf(x**2)) else x
            d = tf.math.log(x + z)
            ###### d = tf.math.acosh(1 + 2 * sqdist / divisor)
            ####d = tf.math.acosh( 1 + sqdist / 2 / tf.math.sqrt(squnorm) / tf.math.sqrt(sqvnorm) )
            # if tf.reduce_any( tf.math.is_nan(d) ):
            #     print( 'we\'ve got a nan in d.')
              
            #     print( 'where: {0}'.format( tf.where(tf.math.is_nan(d)) ) )
            #     print( 'd: {0}'.format( tf.boolean_mask(d, tf.math.is_nan(d)) ) )
            #     print( 's1: {0}'.format( tf.boolean_mask(s1, tf.math.is_nan(d)) ) )
            #     print( 's2: {0}'.format( tf.boolean_mask(s2, tf.math.is_nan(d)) ) )
            #     print( 'sqdist: {0}'.format( tf.boolean_mask(sqdist, tf.math.is_nan(d)) ) )
            #     print( 'squnorm: {0}'.format( tf.boolean_mask(squnorm, tf.math.is_nan(d)) ) )
            #     print( 'sqvnorm: {0}'.format( tf.boolean_mask(sqvnorm, tf.math.is_nan(d)) ) )
                
            #     print( 'x: {0}'.format( tf.boolean_mask(x, tf.math.is_nan(d)) ) )
            #     print( 'z: {0}'.format( tf.boolean_mask(z, tf.math.is_nan(d)) ) )
            #     print( 'd: {0}'.format( tf.boolean_mask(d, tf.math.is_nan(d)) ) )
            #     raise ValueError("we've got a nan in d.")
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
    def __init__(self, vocab_num = 4):
        super().__init__()
        self.vocab = vocab_num
    
    # Decorate with tf.function to fix model saving
    @tf.function
    def call(self, x):
        return tf.one_hot(tf.cast(x, tf.int32), self.vocab)
        
        
        