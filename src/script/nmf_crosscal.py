import numpy as np
from numpy.random import randn, rand
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def crossval_nmf(A, rank, p_holdout=.25, tol=1e-4):
    '''
    Fits NMF while holding out data at random

        Finds W and H that minimize Frobenius norm of (W*H - A) over
        a random subset of the entries in data.


    Parameters
    ----------
    A : ndarray
            m x 8 matrix - m:number of trails
    rank : int
            number of basis vectors
    p_holdout : float
            probability of holding out an entry, expected proportion of data in test set.
    tol: float
            absolute convergence criterion on the root-mean-square-error on training set
    
    Returns
    -------
    W : ndarray
            m x rank matrix
    H : ndarray
            rank x 8 matrix
    train_hist : list
            Root Mean Square Error(RMSE) on training set on each iteration
    test_hist : list
            RMSE on test set on each iteration
    '''

    # m = num observations, n = num emg signals
    m, n = A.shape

    # initialize factor matrices
    W, H = randn(m, rank), randn(rank, n)

    # hold out A at random
    M = rand(m) > p_holdout

    # initial loss
    converged = False
    resid = np.dot(W, H) - A
    train_hist = [np.sqrt(np.mean((resid[M]) ** 2))]
    test_hist = [np.sqrt(np.mean((resid[~M]) ** 2))]

    # initial bias
    delta = 0.000001

    # optimize
    while not converged:
        # Update H
        W_TA = W.T.dot(A)
        W_TWH = W.T.dot(W).dot(H) + delta

        for i in range(rank):
            for j in range(n):
                H[i, j] = H[i, j] * W_TA[i, j] / W_TWH[i, j]

        # Update W
        AH_T = A.dot(H.T)
        WHH_T = W.dot(H).dot(H.T) + delta

        for i in range(m):
            for j in range(rank):
                W[i, j] = W[i, j] * AH_T[i, j] / WHH_T[i, j]

        # recprd train/test error
        resid = np.dot(W, H) - A
        train_hist.append(np.sqrt(np.mean((resid[M]) ** 2)))
        test_hist.append(np.sqrt(np.mean((resid[~M]) ** 2)))
        converged = (train_hist[-2] - train_hist[-1]) < tol

    return W, H, train_hist, test_hist


def VAF(W, H, A):
    """

    Args:
        W: ndarray, m x rank matrix, activation coefficients obtained from nmf
        H: ndarray, rank x 8 matrix, basis vectors obtained from nmf
        A: ndarray, m x 8 matrix, original time-invariant sEMG signal

    Returns:
        global_VAF: float, VAF calculated for the entire A based on the W&H
        local_VAF: 1D array, VAF calculated for each muscle (column) in A based on W&H
    """
    SSE_matrix = np.square(np.dot(W, H) - A)
    SST_matrix = np.square(A)

    global_SSE = np.sum(SSE_matrix)
    global_SST = np.sum(SST_matrix)
    global_VAF = 100 * (1 - global_SSE / global_SST)

    local_SSE = np.sum(SSE_matrix, axis = 0)
    local_SST = np.sum(SST_matrix, axis = 0)
    local_VAF = 100 * (1 - np.divide(local_SSE, local_SST))

    return global_VAF, local_VAF
