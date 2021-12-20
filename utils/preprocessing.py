"""
All preprocessing scripts for NeuroSEEED project
"""

import numpy as np
import pickle

from collections import defaultdict
from multiprocessing import Pool
from functools import partial
from time import time
from itertools import combinations

from Bio import SeqIO
import ete3
import Levenshtein

def fasta_to_numpy(path, lim=None):
    """
    Given a path to a fasta file, create a numpy array of sequences. Key:
        A --> 0
        C --> 1
        T --> 2
        G --> 3

    Args:
    -----
    path:
        String. A path to a FASTA file.

    Returns:
    --------
    X:
        Numpy array of sequences in integer form.
    names:
        List of sequence names. Useful for bookkeeping; necessary for tree-based splitting.
    """

    # Define conversion table
    conversion = defaultdict(lambda: 0)
    conversion['a'] = 1
    conversion['c'] = 2
    conversion['t'] = 3
    conversion['g'] = 4

    # File input
    seqs = []
    names = []
    with open(path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seqs.append(record.seq)
            names.append(record.id)

    # Keep only up to 'lim' of sequences
    seqs = seqs[:lim]
    names = names[:lim]

    # Integer conversion
    embedded_seqs = []
    max_size = 0
    for seq in seqs:
        embedded_seq = [conversion[x] for x in list(seq.lower())]
        embedded_seqs.append(embedded_seq)
        if max_size < len(embedded_seq):
            max_size = len(embedded_seq)

    # End-padding and numpy conversion
    for seq in embedded_seqs:
        pad_size = max_size - len(seq)
        seq += [0] * pad_size
    np_seqs = np.array(embedded_seqs, dtype=int)

    return np_seqs, names



def _dist(X, i):
    """
    Distance function for parallelizing edit_distance_matrix()
    """
    n = len(X)
    d = [Levenshtein.distance(X[i], X[j]) for j in range(n) if i < j]
    return i, d

def edit_distance_matrix(
    X : np.ndarray,
    n_threads : int = 1,
    verbose : bool = False) -> np.ndarray:
    """
    Given a numpy array, return pairwise edit distances.

    Note: some of this code was inspired by the NeuroSEED code, specifically here:
    https://github.com/gcorso/NeuroSEED/blob/master/edit_distance/task/dataset_generator_genomic.py

    Args:
    -----
    X:
        Numpy array (e.g. output from fasta_to_numpy()) of sequences
    n_threads:
        Integer. Number of threads to use for distance computations.
    verbose:
        Boolean. If True, will print out progress updates.

    Returns:
    --------
    y:
        Numpy array of pairwise edit distances.
    """

    n = len(X)
    y = np.zeros((n,n))
    if n_threads > 1:
        pool = Pool(n_threads)
        func = partial(_dist, X)
        vals = pool.imap(func, np.arange(n))
        for i, val in vals:
            m = len(val)
            if m > 0:
                y[i, -m:] = y[-m:, i] = val
            if verbose:
                print(f"Finished row {i} of {n}")
    else:
        for i, seq1 in enumerate(X):
            tick = time()
            for j, seq2 in enumerate(X):
                if i < j:
                    y[i,j] = y[j,i] = Levenshtein.distance(seq1, seq2)
            if verbose:
                tock = time()
                print(f"Finished row {i} of {n}.\tTime taken:{(tock-tick):.03f}")

    return y



def train_test_split_random(
    X : np.ndarray, 
    y : np.ndarray = None,
    names : list = None,
    test_size : float = 0.2,
    val_size : float = 0.2) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Random train-test split. Same as the NeuroSEED paper.

    Args:
    -----
    X:
        Numpy array. X values to split.
    y:
        Numpy array. Optinal Y values to split.
    names:
        List. Node names used for mapping between tree and X values.
    test_size:
        Float. Fraction of data to use for test set.
    val_size:
        Float. Fraction of data to use for validation set.

    Returns:
    --------
    X-values split, optinal Y-values and/or names split

    Raises:
    -------
    TODO
    """

    # Get numbers
    n = len(X)
    indices = np.arange(n)
    n_test = int(n * test_size)
    n_val = int(n * val_size)

    # Get test sets
    test_indices = np.random.choice(indices, n_test, replace=False)
    X_test = X[test_indices]
    indices = np.setdiff1d(indices, test_indices)

    # Get val sets
    val_indices = np.random.choice(indices, n_val, replace=False)
    X_val = X[val_indices]
    indices = np.setdiff1d(indices, test_indices)

    # What's left is training set
    X_train = X[indices]

    output = (X_train, X_test, X_val)
    if y is not None:
        y_train = y[indices, :][:, indices]
        y_test = y[test_indices, :][:, test_indices]
        y_val = y[val_indices, :][:, val_indices]
        output = (output, (y_train, y_test, y_val))
    if names:
        names_train = names[indices]
        names_test = names[test_indices]
        names_val = names[val_indices]
        output = output, (names_train, names_test, names_val)
    return output

def _get_subtrees(tree, names, size, split_depth, limit_subtree_size) -> (list, ete3.Tree):
    """
    Helper function for train_test_split_tree()
    """
    out = []
    while len(out) < size:
        leaf = np.random.choice(tree.get_leaves())
        parent = leaf
        for i in range(split_depth):
            parent = parent.up
        if not limit_subtree_size or len(parent.get_leaves()) <= 2**split_depth:
            subtree = parent.detach()
            children = [leaf.name for leaf in subtree.get_leaves()]
            children = [x for x in children if x in names]
            out += children
    return out, tree

def train_test_split_tree(
    X : np.ndarray, 
    y : np.ndarray = None,
    names : list = None,
    test_size : float = 0.2,
    val_size : float = 0.2, 
    tree_path : str = None,
    split_depth : int = 2,
    drop_nontree_seqs : bool = False,
    limit_subtree_size : bool = True) -> ((np.ndarray, np.ndarray, np.ndarray), (list, list, list)):
    """
    Clade-based train-test split.

    Args:
    -----
    X:
        Numpy array. X values to split.
    y:
        Numpy array. Optinal Y values to split.
    names:
        List. Node names used for mapping between tree and X values.
    test_size:
        Float. Fraction of data to use for test set.
    val_size:
        Float. Fraction of data to use for validation set.
    tree_path:
        String. A path to a Newick tree specifying the phylogenetic relationship between sequences.
    split_depth:
        Integer. Depth of subtrees used int train-test split.
    drop_nontree_seqs:
        Boolean. This will limit X to sequences that are in the provided tree.
    limit_subtree_size:
        Boolean. This will ensure subtrees are no larger than 2**split_depth.

    Returns:
    --------
    Split X-values, optional split Y-values, and split names.

    Raises:
    -------
    TODO
    """

    # Input validations
    if not names:
        raise ValueError(f"Need names to use tree-based split!")
    if not tree_path:
        raise ValueError(f"Need tree path for tree-based split!")

    # Get numbers
    n = len(X)
    indices = np.arange(n)

    # Get tree; prune to relevant leaves
    tree = ete3.Tree(tree_path, quoted_node_names=True, format=1)
    if drop_nontree_seqs:
        leaves = [leaf.name for leaf in tree.get_leaves()]
        keep_idx = [i for i in indices if names[i] in leaves]
        names = [names[i] for i in keep_idx]
        X = X[keep_idx]
        if y:
            y = y[keep_idx,:][:,keep_idx]
        n = len(X)
        indices = np.arange(n)
    tree.prune(names, preserve_branch_length=True)

    # Compute numbers after tree-splitting has happened
    n_test = int(n * test_size)
    n_val = int(n * val_size)

    # Get test and validation sets:
    test_names, tree = _get_subtrees(tree, names, test_size, split_depth, limit_subtree_size)
    val_names, tree = _get_subtrees(tree, names, val_size, split_depth, limit_subtree_size)

    # Get training set
    train_names = [leaf.name for leaf in tree.get_leaves()]
    train_names = [x for x in train_names if x in names]

    # Convert to indices
    train_indices = [i for i in indices if names[i] in train_names]
    test_indices = [i for i in indices if names[i] in test_names]
    val_indices = [i for i in indices if names[i] in val_names]
    X_train = X[train_indices]
    X_test = X[test_indices]
    X_val = X[val_indices]
    output = (X_train, X_test, X_val)
    if y is not None:
        y_train = y[train_indices, :][:, train_indices]
        y_test = y[test_indices, :][:, test_indices]
        y_val = y[val_indices, :][:, val_indices]
        output = (output, (y_train, y_test, y_val))

    return output, (train_names, test_names, val_names)



def _get_clusters(y, indices, size, split_depth) -> (np.ndarray, np.ndarray):
    """
    Helper function for train_test_split_distance()
    """
    out = np.array([])
    while len(out) < size:
        row_idx = np.random.choice(indices)
        row = y[row_idx]
        n_relatives = 2 ** split_depth # Don't need to do -1 since we include the sequence itself
        relatives = np.argpartition(row, n_relatives)[:n_relatives]
        relatives = np.intersect1d(relatives, indices)
        out = np.union1d(relatives, out)
        indices = np.setdiff1d(indices, out)
    return out.astype(int), indices.astype(int)



def train_test_split_distance(
    X : np.ndarray,
    y : np.ndarray = None,
    names : list = None,
    test_size : float = 0.2,
    val_size : float = 0.2,
    split_depth : int = 2) -> ((np.ndarray, np.ndarray, np.ndarray), (np.ndarray, np.ndarray, np.ndarray)):
    """
    Distance-based train-test split.

    Args:
    -----
    X:
        Numpy array. Sequences to split.
    y:
        Numpy array. Y-values to split. Required for distance-based splitting.
    names:
        List. Optional list of names to split.
    test_size:
        Float. Fraction of sequences to use for test set.
    val_size:
        Float. Fraction of sequences to use for validation set.
    split_depth:
        Integer. Height of subtrees to extract.

    Returns:
    --------
    X and Y split into training/test/validation sets.

    Raises:
    -------
    TODO
    """
    # Get numbers
    n = len(X)
    indices = np.arange(n)
    n_test = int(n * test_size)
    n_val = int(n * val_size)

    # Get test data
    test_indices, indices = _get_clusters(y, indices, n_test, split_depth)
    val_indices, indices = _get_clusters(y, indices, n_val, split_depth)

    # Get training set
    X_train = X[indices]
    X_test = X[test_indices]
    X_val = X[val_indices]

    # Output steps
    output = (X_train, X_test, X_val)
    if y is not None:
        y_train = y[indices, :][:, indices]
        y_test = y[test_indices, :][:, test_indices]
        y_val = y[val_indices, :][:, val_indices]
        output = (output, (y_train, y_test, y_val))
    if names:
        names_train = names[indices]
        names_test = names[test_indices]
        names_val = names[val_indices]
        output = output, (names_train, names_test, names_val)
    return output



def process_seqs(
    seqs_path : str,
    tree_path : str = None,
    train_test_split : str = 'random',
    split_depth : int = 2,
    test_size : float =  0.2,
    val_size : float = 0.2,
    lim : int = None,
    drop_nontree_seqs : bool = False,
    n_threads : int = 1,
    save_y : str = False,
    load_y : str = False,
    verbose : bool = True) -> ((np.array, np.array, np.array), (np.array, np.array, np.array)):
    """
    Given a path to a fasta file, generate X and Y inputs

    Args:
    ----
    seqs_path:
        String. A path to a FASTA file containing sequences.
    tree_path:
        String. A path to a Newick tree specifying the phylogenetic relationship between sequences.
    train_test_split:
        String. One of {'random', 'tree', 'distance'}. Behaves as follows:
        * 'random': train-test split samples leaves randomly.
        * 'distance': train-test split will sample subtrees of nearest neighbors of size 2^(split_depth).
        * 'tree': train-test split will sample subtrees of depth split_depth from phylogenetic tree.
    split_depth:
        Integer. Depth of subtrees used int train-test split. Ignored if train_test_split='random'.
    test_size:
        Float. Fraction of data to use for test set.
    val_size:
        Float. Fraction of data to use for validation set.
    lim:
        Integer. Cutoff for number of input sequences to read in.
    drop_nontree_seqs:
        Boolean. If train_test_split = 'tree', this will limit X to sequences that are in the provided tree.
    n_threads:
        Integer. Number of threads used for edit distance calculation.
    save_y:
        String. Path to save y matrices.
    load_y:
        String. Path to load y matrices.
    verbose:
        Boolean. Prints out 
    
    Returns:
    --------
    A nested tuple of ((X_train, X_test, X_val), (y_train, y_test, y_val)). Dimensions should be as follows:
    X_train:    (N * (1 - test_size) * (1 - val_size), sequence_length)
    X_test:     (N * test_size, sequence_length)
    X_val:      (N * test_size, sequence_length)
    y_train:    (N * (1 - test_size) * (1 - val_size), N * (1 - test_size) * (1 - val_size))
    y_test:     (N * test_size, N * test_size)
    y_val:      (N * val_size, N * val_size)

    Raises:
    -------
    TODO
    """

    # Read inputs
    if verbose:
        tick = time()
        print("Reading inputs...")
    X, names = fasta_to_numpy(seqs_path, lim=lim)
    if verbose:
        tock = time()
        print(f"\tDone in {(tock - tick):.03f} seconds")
        print(f"\tShape of X: {X.shape}")

    # Need distances first if we do train-test split by distance
    if train_test_split == "distance":
        if load_y:
            if verbose:
                print(f"Loading distances from {load_y}")
            with open(load_y, "rb") as f:
                y = pickle.load(f)
        else:
            if verbose:
                tick = time()
                print("Computing distances...")
            y = edit_distance_matrix(X, verbose=verbose, n_threads=n_threads)
            if save_y:
                with open(save_y, "wb") as f:
                    pickle.dump(y, f)
            if verbose:
                tock = time()
                print(f"\tDone in {(tock - tick):.03f} seconds")
                print(f"\tShape of y: {y.shape}")

    # Train/test/validation split on X
    if verbose:
        tick = time()
        print("Splitting X values...")
    if train_test_split == 'none':
        X_train, X_test, X_val = X, None, None
    elif train_test_split == 'random':
        X_train, X_test, X_val = train_test_split_random(X, test_size=test_size, val_size=val_size)
    elif train_test_split == 'tree':
        (X_train, X_test, X_val), (names_train, names_test, names_val) = train_test_split_tree(X,
            names=names,
            test_size=test_size, 
            val_size=val_size, 
            tree_path=tree_path, 
            split_depth=split_depth,
            drop_nontree_seqs=drop_nontree_seqs
        )
    elif train_test_split == 'distance':
        (X_train, X_test, X_val), (y_train, y_test, y_val), = train_test_split_distance(X, y,
            test_size=test_size, 
            val_size=val_size,
            split_depth=split_depth
        )
    else:
        raise ValueError(f"Train-test split strategy '{train_test_split}' not recognized")
    if verbose:
        tock = time()
        print(f"\tDone in {(tock - tick):.03f} seconds")
        print(f"\tShapes of data: {X_train.shape}, {X_test.shape}, {X_val.shape}")

    # Compute y values after train-test split for faster computations
    if train_test_split != "distance":
        if load_y:
            with open(load_y, "rb") as f:
                y_train, y_test, y_val = pickle.load(f)

        else:
            if verbose:
                tick = time()
                print("Getting edit distances for y_train...")
            y_train = edit_distance_matrix(X_train, verbose=verbose, n_threads=n_threads)
            if verbose:
                tock = time()
                print(f"\tDone in {(tock - tick):.03f} seconds")
                print(f"\tShape of y_train: {y_train.shape}")

            if verbose:
                tick = time()
                print("Getting edit distances for y_test...")
            y_test = edit_distance_matrix(X_test, verbose=verbose, n_threads=n_threads)
            if verbose:
                tock = time()
                print(f"\tDone in {(tock - tick):.03f} seconds")
                print(f"\tShape of y_test: {y_test.shape}")

            if verbose:
                tick = time()
                print("Getting edit distances for y_val...")
            y_val = edit_distance_matrix(X_val, verbose=verbose, n_threads=n_threads)
            if verbose:
                tock = time()
                print(f"\tDone in {(tock - tick):.03f} seconds")
                print(f"\tShape of y_val: {y_val.shape}")

            if save_y:
                with open(save_y, "wb") as f:
                    pickle.dump((y_train, y_test, y_val), f)

    return ((X_train, X_test, X_val), (y_train, y_test, y_val))
