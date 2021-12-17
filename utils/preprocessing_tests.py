"""
Basic unit tests for preprocessing scripts
"""
from preprocessing import *

# Test definitions
def fasta_to_numpy_tests():
    path = '/Users/phil/Documents/Columbia/year2/4040/e4040-2021Fall-Project-BIOM-rr3415-pc2946-hs3164/data/test_seqs.fasta'
    X, names = fasta_to_numpy(path)
    print(f"Shape of output: {X.shape}")

def edit_distance_matrix_test():
    path = '/Users/phil/Documents/Columbia/year2/4040/e4040-2021Fall-Project-BIOM-rr3415-pc2946-hs3164/data/test_seqs.fasta'
    X, _ = fasta_to_numpy(path)
    print(f"Shape of X: {X.shape}")

    # Downsample
    X = X[:100]

    y = edit_distance_matrix(X, verbose=True)
    print(f"Shape of y: {y.shape}")

def train_test_split_random_test():
    X = 10 * np.random.rand(100, 100)
    X = X.astype(int)
    y = edit_distance_matrix(X)

    ((X, Xt, Xv), (y, yt, yv)) = train_test_split_random(X, y, test_size=0.2, val_size=0.2)

    print(X.shape, Xt.shape, Xv.shape, y.shape, yt.shape, yv.shape)

def train_test_split_random_test_no_y():
    X = 10 * np.random.rand(100, 100)
    X = X.astype(int)

    X, Xt, Xv = train_test_split_random(X, test_size=0.2, val_size=0.2)

    print(X.shape, Xt.shape, Xv.shape)

def pipeline_test_1():
    path = '/Users/phil/Documents/Columbia/year2/4040/e4040-2021Fall-Project-BIOM-rr3415-pc2946-hs3164/data/test_seqs.fasta'
    process_seqs(path)

def pipeline_test_tree():
    path = '/Users/phil/Documents/Columbia/year2/4040/e4040-2021Fall-Project-BIOM-rr3415-pc2946-hs3164/data/gg_12_10.fasta'
    process_seqs(path, drop_nontree_seqs = True, split_depth=2, lim=1000, train_test_split='tree', tree_path='/Users/phil/Documents/Columbia/year2/4040/e4040-2021Fall-Project-BIOM-rr3415-pc2946-hs3164/data/gg_12_10_otus_99_annotated.tree')

def pipeline_test_distance():
    # path = '/Users/phil/Documents/Columbia/year2/4040/e4040-2021Fall-Project-BIOM-rr3415-pc2946-hs3164/data/test_seqs.fasta'
    path = './data/test_seqs.fasta'
    process_seqs(path, train_test_split='random', n_threads=12)

# Test calls
if __name__ == '__main__':
    # fasta_to_numpy_tests()
    # edit_distance_matrix_test()
    # train_test_split_random_test()
    # train_test_split_random_test_no_y()
    # pipeline_test_1()
    # pipeline_test_tree()
    pipeline_test_distance()