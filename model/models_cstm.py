import numpy as np
import tensorflow as tf

class SiameseModel(tf.keras.models.Model):
    def __init__(self, siamese_network):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    
    def call(self, input):
        return self.siamese_network(input)
    
    def _loss(self, data):
        ((s1, s2), y) = data
        embed_dist = self.siamese_network([s1, s2])
        return (y - embed_dist)**2 # squared error
    
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

def get_embedding_model():
    # Basic dense NN
    embedding = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=4, output_dim=2, input_shape=(152,)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
    ])

    return embedding