import numpy as np
import tensorflow as tf
import itertools

class SequenceDistDataGenerator(tf.keras.utils.Sequence):
    """
    Generates processed X and Y data, where X data represents pairs of sequences, as a list of two arrays of size (batch_size, sequence_dim)
    and Y data represents the distances between the sequences in each pair, of dimension (batch_size, 1)

    Adapted from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self, x_seq, dist_mat, batch_size=32, shuffle=True, **kwargs ):
        'Initialization'
        self.x_seq = x_seq
        self.dist_mat = dist_mat
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        #print('batch size: {0}'.format(self.batch_size))
        #print('shuffle: {0}'.format(self.shuffle))
        #print('kwargs: {0}'.format(kwargs))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate (paired) indexes for the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(batch_indexes)
        return X, y

    def on_epoch_end(self):
        'Updates complete list of indexes after each epoch'
        # Note that indexes are paired here - basically, all the combinations of sample indices
        self.indexes = list( itertools.combinations( np.arange(len(self.x_seq)), 2 ) )
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indexes):
        'Generates data containing batch_size samples' 
        # X : list of size 2, i.e. list([X0, X1]), containing the pairs of samples
        # X0: array of dimension (batch_size, sequence_dim)
        # X1: array of dimension (batch_size, sequence_dim)
        # y: dist between the pairs of samples in the corresponding entries of X0 and X1, array of length batch_size
        # Initialization
        X = list()
        X.append(np.empty((self.batch_size, self.x_seq.shape[1])))
        X.append(np.empty((self.batch_size, self.x_seq.shape[1])))
        y = np.empty((self.batch_size), dtype='float32')

        # Generate data
        for i in range(len(batch_indexes)):
            # Store sample pairings
            X[0][i,:] = self.x_seq[batch_indexes[i][0],:]
            X[1][i,:] = self.x_seq[batch_indexes[i][1],:]
            # Store distance between the pair
            y[i] = self.dist_mat[batch_indexes[i][0], batch_indexes[i][1]]

        return X, y