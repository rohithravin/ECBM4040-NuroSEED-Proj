import numpy as np
import tensorflow as tf
from model.layer import OneHotEncodingLayer

class SiameseModel(tf.keras.models.Model):
    def __init__(self, siamese_network):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    
    # Decorate with tf.function to fix model saving
    @tf.function
    def call(self, input):
        return self.siamese_network(input)
    
    def _loss(self, data):
        ((s1, s2), y) = data
        embed_dist = self.siamese_network([s1, s2])
        loss_val = (y - embed_dist)**2 # squared error
        #if tf.reduce_any( tf.math.is_nan(loss_val) ):
        #    raise ValueError( 'Loss has become nan.' )
        return loss_val
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._loss(data)
        grad = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.siamese_network.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss" : self.loss_tracker.result()}
    
    def test_step(self, data):
        loss = self._loss(data)
        self.loss_tracker.update_state(loss)
        return {"loss" : self.loss_tracker.result()}
    
    @property
    def metrics(self):
        return [self.loss_tracker]

def get_embedding_model(in_dim=152, out_dim=128, model_choice='LINEAR', act_func='relu', dropout=0.0, 
                        mlp_num_units_hidden=128, cnn_num_filters=4):
    if model_choice=='LINEAR':
        # Linear dense output layer
        embedding = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(in_dim,)),
            tf.keras.layers.Dropout(dropout),
            OneHotEncodingLayer(4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(out_dim, activation=None),
        ])
    elif model_choice=='MLP':
        # Dense with non-linear hidden layers
        embedding = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(in_dim,)),
            tf.keras.layers.Dropout(dropout),
            OneHotEncodingLayer(4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(mlp_num_units_hidden, activation=act_func),
            tf.keras.layers.Dense(mlp_num_units_hidden, activation=act_func),
            tf.keras.layers.Dense(out_dim, activation=None),
        ])  
    elif model_choice=='CNN':
        # CNN model
        embedding = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(in_dim,)),
            tf.keras.layers.Dropout(dropout),
            OneHotEncodingLayer(4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Reshape((in_dim*4,1)),
            tf.keras.layers.Conv1D(cnn_num_filters, 3, padding='same', activation=act_func),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.AveragePooling1D(2),
            tf.keras.layers.Conv1D(cnn_num_filters, 3, padding='same', activation=act_func),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.AveragePooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(out_dim, activation=None)
        ])  

    return embedding