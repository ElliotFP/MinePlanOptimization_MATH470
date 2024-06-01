# This is a solver for problems that take the form of block angular matrices.
# This specific solver assumes that each block has a single constraint, and that
# each block's constraint has the same number of variables.

import numpy as np

def block_angular_solve(A, b):
    """
    Input:
        A: A list of numpy arrays, where each array is a block in the block angular matrix. Corresponds to the A matrix in the linear program.
        b: A numpy array that corresponds to the constraints
    Output:
        A_prime: A simplified version of A, using the particular block angular structure of the matrix.
        b_prime: A simplified version of b, using the particular block angular structure of the matrix.
    """