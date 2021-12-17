"""
Utility script for processing data. Usage is as follows:

    python utils/compute_distances.py [path/to/seqs] [path/to/y_vals] [path/to/output] [depth]
"""

from preprocessing import process_seqs
import sys
import os
from multiprocessing import cpu_count
import pickle

if __name__ == '__main__':
    # Parse arguments
    seqs_path = sys.argv[1]
    y_path = sys.argv[2]
    data_path = sys.argv[3]
    depth = sys.argv[4]
    print(seqs_path, y_path)

    # Parse additional flags


    if os.path.exists(y_path):
        data = process_seqs(
            seqs_path=seqs_path,
            train_test_split='distance',
            split_depth=depth,
            test_size=0.2,
            val_size=0.2,
            load_y=y_path,
            verbose=True
        )
    else:
        n_threads = cpu_count()
        print(f"Running edit distance computation with {n_threads} threads")
        data = process_seqs(
            seqs_path=seqs_path,
            train_test_split='distance',
            split_depth=depth,
            test_size=0.2,
            val_size=0.2,
            save_y=y_path,
            n_threads=n_threads,
            verbose=True
        )
    with open(data_path, "wb") as f:
        pickle.dump(data, f)