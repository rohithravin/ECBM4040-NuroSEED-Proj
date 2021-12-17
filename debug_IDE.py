'''
Python script for debugging in an IDE
''' 


######## imports and initializations

import numpy as np
import tensorflow as tf
import pickle
from matplotlib import pyplot as plt

import shutil
import os


from model.models_cstm import get_embedding_model
from model.train_model import train_siamese_model

DISTANCE_METRICS = {
    'EUCLIDEAN': 'euclidean',
    'HYPERBOLIC': 'hyperbolic',
    'MANHATTAN': 'manhattan',
    'SQUARE': 'square',
    'COSINE': 'cosine'
}

# set random number seeds for reproducible results
np.random.seed(1)
tf.random.set_seed(1)


############### load data
cwd = os.getcwd()
# Load QIITA dataset.
(x_dat_dict, y_dat_dict) = pickle.load(open(f"{cwd}/data/qiita/qiita_numpy.pkl", "rb"))
(X_train, X_test, X_val) = (x_dat_dict['train'], x_dat_dict['test'], x_dat_dict['val'])
(y_train, y_test, y_val) = (y_dat_dict['train'], y_dat_dict['test'], y_dat_dict['val'])

############## train siamese model
# Train and Test Siamese Model
embedding = get_embedding_model()
dat_lim = len(X_train)
data = ((X_train[:dat_lim], X_test[:dat_lim], X_val[:dat_lim]), (y_train[:dat_lim,:dat_lim], y_test[:dat_lim], y_val[:dat_lim,:dat_lim]))
dist = DISTANCE_METRICS['HYPERBOLIC']

model, score, history = train_siamese_model(data, embedding, dist , batch_size=256, epochs=20)

#print score
print(f'Score for Siamese Model using {dist} distance: {score}')


################ plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()