import numpy as np
import tensorflow as tf

from model.layer import DistanceLayer
from model.models_cstm import SiameseModel
from model.generator import SequenceDistDataGenerator

def train_siamese_model(
    data : tuple,
    embedding_model : tf.keras.Model,
    distance_metric : str,
    optimizer : tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(1e-5, clipnorm=1),
    **kwargs) -> tf.keras.Model:
    """
    Given training/validation/test data, an embedding model, and a distance metric, perform the NeuroSEED edit distance
    reconstruction task.

    Args:
    -----
    data:
        Tuple. Should be ((X_train, X_test, X_val),(y_train, y_test, y_val)).
    embedding_model:
        Keras model used for generating embeddings.
    distance_metric:
        One of {'euclidean', 'hyperbolic', 'manhattan'}
    optimizer:
        Keras model optimizer.
    **kwargs:
        Passed to model training.
    
    Returns:
    --------
    A trained model minimizing MSE between embedding distance and true distance encoded in Y values.

    Raises:
    -------
    TODO
    """

    # Data loading
    ((X_train, X_test, X_val),(y_train, y_test, y_val)) = data

    # Input validation
    d1 = X_train.shape[1]
    d2 = X_test.shape[1]
    d3 = X_val.shape[1]

    if d1 != d2 or d2 != d3:
        raise ValueError(f"Dimension mismatch: {d1}, {d2}, {d3} should all be equal")

    # Generators from data
    training_generator = SequenceDistDataGenerator( X_train, y_train, **kwargs )
    validation_generator = SequenceDistDataGenerator( X_val, y_val, **kwargs )
    testing_generator = SequenceDistDataGenerator( X_test, y_test, **kwargs )

    # Model definitions
    in1 = tf.keras.layers.Input(name="sequence1", shape=(d1,))
    in2 = tf.keras.layers.Input(name="sequence2", shape=(d1,))

    distance = DistanceLayer(metric=distance_metric, dynamic=True)(
        embedding_model(in1), 
        embedding_model(in2)
    )

    siamese_network = tf.keras.models.Model(
        inputs=[in1, in2],
        outputs=distance
    )

    model = SiameseModel(siamese_network) # Depends on SiameseModel class, which we can define elsewhere
    model.compile(optimizer=optimizer, run_eagerly=True) # run_eagerly is not necessary, but useful for debugging

    # Training
    history = model.fit(training_generator, validation_data=validation_generator, **kwargs)

    # Evaluate
    score = model.evaluate(testing_generator)

    return model, score, history