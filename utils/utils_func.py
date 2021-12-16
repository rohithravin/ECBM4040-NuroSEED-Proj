import numpy as np

def make_pairs(x,y):
    """
    Given an X and Y array such that X is (N, M) and Y is (N, N), returns processed X' and Y' such that
    X is (2, N^2, M) and Y is (N^2, 1) encoding pairs of sequences and their distances.

    TODO: verify these dimensions
    TODO: replace this with a data loader
    """
    out_x = [[],[]]
    out_y = []
    # TODO: vectorize (and replace with dataloader)
    for i, si in enumerate(x):
        for j, sj in enumerate(x):
            out_x[0].append(si)
            out_x[1].append(sj)
            out_y.append(y[i,j])
    out_x = [np.array(dataset) for dataset in out_x]
    out_y = np.array(out_y)
    return out_x, out_y