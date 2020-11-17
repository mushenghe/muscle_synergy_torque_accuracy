import numpy as np

'''
matrix factorization algorithms
'''

def multiplication_update(A, k, thresh=1e-4, num_iter=100, init_W=None, init_H=None, print_enabled=False):

    '''
    Run multiplicative updates to perform nonnegative matrix factorization on A.
    Return matrices W, H such that A = WH.

    Parameters:
        A: ndarray
            - m by n matrix to factorize
        k: int
            - integer specifying the column length of W / the row length of H
            - the resulting matrices W, H will have sizes of m by k and k by n, respectively
        delta: float
            - float that will be added to the numerators of the update rules
            - necessary to avoid division by zero problems
        num_iter: int
            - number of iterations for the multiplicative updates algorithm
        init_W: ndarray
            - m by k matrix for the initial W
        init_H: ndarray
            - k by n matrix for the initial H
        print_enabled: boolean
            - if ture, output print statements

    Returns:
        W: ndarray
            - m by k matrix where k = dim
        H: ndarray
            - k by n matrix where k = dim
    '''

    # print('Applying multiplicative updates on the input matrix...')

    if print_enabled:
        print('---------------------------------------------------------------------')
        print('Frobenius norm ||A - WH||_F')
        print('')

    # Initialize W and H
    if init_W is None:
        W = np.random.rand(np.size(A, 0), k)
    else:
        W = init_W

    if init_H is None:
        H = np.random.rand(k, np.size(A, 1))
    else:
        H = init_H

    delta = 0.000001
    itt = 1
    below_thresh = False

    A = np.array(A)
    W = np.array(W)
    H = np.array(H)
    resid = np.dot(W, H) - A

    # sse_hist = []
    # SSE_hist = []
    error_hist = [np.sqrt(np.mean((resid)**2))]
    # Decompose the input matrix
    while not below_thresh:

        # Update H
        W_TA = W.T.dot(A)
        W_TWH = W.T.dot(W).dot(H) + delta

        for i in range(np.size(H, 0)):
            for j in range(np.size(H, 1)):
                H[i, j] = H[i, j] * W_TA[i, j] / W_TWH[i, j]

        # Update W
        AH_T = A.dot(H.T)
        WHH_T = W.dot(H).dot(H.T) + delta

        for i in range(np.size(W, 0)):
            for j in range(np.size(W, 1)):
                W[i, j] = W[i, j] * AH_T[i, j] / WHH_T[i, j]

        
        resid = np.dot(W, H) - A
        # SE = np.square(resid)

        # SSE_hist.append(np.sum(SE))  #scaler
        # sse_hist.append(np.sum(SE,axis = 0)) #(1,8)
        error_hist.append(np.sqrt(np.mean((resid)**2)))

        below_thresh = (error_hist[-2]-error_hist[-1]) < thresh


        error = np.linalg.norm(A - np.dot(W, H), ord=2)
        if error < thresh:
            below_thresh = True
        itt += 1


        if print_enabled:
            frob_norm = np.linalg.norm(A - np.dot(W, H), 'fro')
            print("iteration " + str(n + 1) + ": " + str(frob_norm))

    return W, H


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
