import numpy as np
from numpy.random import randn, rand
from scipy.optimize import minimize
import matplotlib.pyplot as plt
def censored_least_squares(W, H, A, M, options=dict(maxiter=20), **kwargs):


    """Approximately solves least-squares with missing/censored data by L-BFGS-B

        Updates X to minimize Frobenius norm of M .* (A*X - B), where
        M is a masking matrix (m x n filled with zeros and ones), A and
        B are constant matrices.

    Parameters
    ----------
    W : ndarray
            m x k matrix
    H : ndarray
            k x n matrix, initial guess for X
    A : ndarray
            m x n matrix
    M : ndarray
            m x n matrix, filled with zeros and ones
    options : dict
            optimization options passed to scipy.optimize.minimize

    Note: additional keyword arguments are passed to scipy.optimize.minimize

    Returns
    -------
    result : OptimizeResult
            returned by scipy.optimize.minimize
    """
    k = W.shape[1]
    n = A.shape[1]

    def fg(x):
        X = x.reshape(k, n)
        resid = np.dot(W, X) - A
        f = 0.5*np.sum(resid[M]**2)
        g = np.dot(W.T, (M * resid))

        return f, g.ravel()

    return minimize(fg, H.ravel(), method='L-BFGS-B', jac=True, options=options, **kwargs)

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
    Mask = list(rand(m) > p_holdout)
    M = np.array([[np.repeat(Mask[i], n)] for i in range(m)]).reshape(m,n)

    # initial loss
    converged = False
    resid = np.dot(W, H) - A
    train_hist = [np.sqrt(np.mean((resid[M]) ** 2))]
    test_hist = [np.sqrt(np.mean((resid[~M]) ** 2))]

    # impose nonnegativity
    bounds_H = [(0, None) for _ in range(H.size)] 
    bounds_W = [(0, None) for _ in range(W.size)]

  
    # optimize
    while not converged:
        r = censored_least_squares(W, H, A, M, bounds = bounds_H)
        H = r.x.reshape(rank, n)

        # update W
        r = censored_least_squares(H.T, W.T, A.T, M.T, bounds = bounds_W)
        W = r.x.reshape(rank, m).T

        # recorerd train/test error
        resid = np.dot(W, H) - A

        train_hist.append(np.sqrt(np.mean((resid[M])**2)))
        test_hist.append(np.sqrt(np.mean((resid[~M])**2)))
        converged = (train_hist[-2]-train_hist[-1]) < tol
        

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
