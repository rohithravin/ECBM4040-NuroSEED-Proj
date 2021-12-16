import numpy as np
import tensorflow as tf
import itertools

class SequenceDistDataGenerator(tf.keras.utils.Sequence):
    """
    Processes and generates batches of X and Y data on the fly, where X data represents pairs of sequences, as a list of two arrays of size (batch_size, sequence_dim)
    and Y data represents the distances between the sequences in each pair, of dimension (batch_size, 1)

    In order to write this code, we needed to first figure out how to write a generic data generator, for which
    we consulted this link: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    
    Therefore, we started from the generic data generator code at that link, and then adapted that generic code to fit the very
    unique needs of our problem, where we need to generate TWO paired input sequences (and don't need to use things like data file IDs, etc.)
    """
    def __init__(self, x_seq, dist_mat, batch_size=32, shuffle=True, **kwargs ):
        ''' 
        Initialize the SequenceDistDataGenerator class.
        
        Inputs:
            x_seq      (np array): array of input sequences (original X matrix of all samples); dim = ( n, sequence_dim )
            dist_mat   (np array): distance matrix representing distance between input sequences (ordered the same as X); dim = ( n, n )
            batch_size (int): batch size (generic data generator parameter)
            shuffle    (bool): indicator of whether to shuffle samples after each epoch (generic data generator parameter)
        '''
        self.x_seq = x_seq
        self.dist_mat = dist_mat
        self.batch_size = batch_size 
        self.shuffle = shuffle
        self.on_epoch_end()
        #print('batch size: {0}'.format(self.batch_size))
        #print('shuffle: {0}'.format(self.shuffle))
        #print('kwargs: {0}'.format(kwargs))

    def __len__(self):
        '''
        Compute number of batches per epoch.
        
        NOTE: this is a generic data generator function; unchanged from implementation in https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        '''
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        '''
        Generate one batch of data.
        
        NOTE: this is a generic data generator function; unchanged from implementation in https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        
        Inputs:
            index (int): batch index
        '''
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(batch_indexes)
        return X, y

    def on_epoch_end(self):
        '''
        Update complete list of index PAIRS after each epoch (and shuffle if indicated).
        
        Note: here we make an update such that we return all COMBINATIONS of paired indexes. 
        This is to address the specific needs of our problem where we are interested in distances between sample sequence pairs.
        '''
        # compute all combinations
        self.indexes = list( itertools.combinations( np.arange(len(self.x_seq)), 2 ) )
        # shuffle if needed
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indexes):
        '''
        Generate data containing batch_size samples.
        
        Inputs:
            batch_indexes (list): list of combinations of samples
            
        Returns:
            X : list of size 2, i.e. list([X0, X1]), containing the pairs of samples
                X0: array of dimension (batch_size, sequence_dim)
                X1: array of dimension (batch_size, sequence_dim)
            y: dist between the pairs of samples in the corresponding entries of X0 and X1, array of length batch_size
        '''
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