import matplotlib.pyplot as plt
import numpy as np

# Step1: Load baseline EMG data, select datas when state = 1 and compute the mean for each EMG signal
# Step2: Extract data for state 4 and 5 from MatchingTask:
#        For state 4: For the last piece of data whose state is 4, take 500 pieces of data 1 second ago
#        For state 5: For the last piece of data whose state is 5, take 250 pieces of data 0.25 second ago and 0.25 second later
# Step3: Compute the mean for all data points and get 2 8x1 vector for each trail
# Step4: subtract the noise vector from the data vector
# Step5: Use NMF to factorize the matrix

def first_last_index(arr):
    prev = arr[0]
    first = arr[-1]
    for num in arr[1:]:
        if (prev + 1) != num:
            first = prev
            break
        prev += 1
    return first

def multiplication_update(A, k, thresh = 0.01, num_iter = 100, init_W=None, init_H=None, print_enabled=False):
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

    print('Applying multiplicative updates on the input matrix...')

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
    # Decompose the input matrix
    while not below_thresh and itt <= num_iter:

        # Update H
        W_TA = W.T.dot(A)
        W_TWH = W.T.dot(W).dot(H) + delta

        for i in range(np.size(H, 0)):
            for j in range(np.size(H, 1)):
                H[i, j] = H[i, j] * W_TA[i, j] / W_TWH[i, j]

        # Update W
        AH_T = A.dot(H.T)
        WHH_T =  W.dot(H).dot(H.T) + delta

        for i in range(np.size(W, 0)):
            for j in range(np.size(W, 1)):
                W[i, j] = W[i, j] * AH_T[i, j] / WHH_T[i, j]
        
        error = np.linalg.norm(A - np.dot(W,H), ord=2)  
        if error < thresh:
            below_thresh = True 
        itt +=1

        if print_enabled:
            frob_norm = np.linalg.norm(A - np.dot(W ,H), 'fro')
            print("iteration " + str(n + 1) + ": " + str(frob_norm))

    return W, H

# Step1 Load baseline EMG data, select datas when state = 1 and compute the mean for each EMG signal
baselineData = np.loadtxt("/home/mushenghe/Desktop/final_project/muscle_synergy/data/BaselineEMG/set01_trial01.txt")
target_baseline = baselineData[np.where(baselineData[:,1] < 1.9), 10:18].reshape((6401, 8))
baseline = np.mean(target_baseline, axis = 0)


# Step 2 Extract data for state 4 and 5 from MatchingTask:
SET1_TRAILS = ['set01_trial01.txt','set01_trial02.txt','set01_trial03.txt','set01_trial04.txt','set01_trial05.txt','set01_trial06.txt', \
    'set01_trial07.txt', 'set01_trial08.txt', 'set01_trial09.txt' ,'set01_trial10.txt']
SEG_STATE4 = []
SEG_STATE5 = []

for trail_index in range (0,10):
    trail_data = np.loadtxt('/home/mushenghe/Desktop/final_project/muscle_synergy/data/MatchingTask/' + SET1_TRAILS[trail_index])
    rows,columns = trail_data.shape
    # reverse the array to find the last data piece
    reversed_data = np.flip(trail_data,0) 
    reversed_state = reversed_data[:,1]  
    indexof_lpl4 = np.nonzero(reversed_state == 4.00000)[0]
    indexof_lpl5 = np.nonzero(reversed_state == 5.00000)[0]
    # index of the last piece of data whose state is 4/5 in the reversed array
    indexof_lpf4 = first_last_index(indexof_lpl4)
    indexof_lpf5 = first_last_index(indexof_lpl5)
    # index of the last piece of data whose state is 4/5 in the reversed array
    # seg_state4 = np.mean(reversed_data[indexof_lpf4 + 1000:indexof_lpf4 + 1500,10:18], axis = 0) - noise
    # seg_state5 = np.mean(reversed_data[indexof_lpf5 - 250:indexof_lpf5 + 250,10:18], axis = 0) - noise
    seg_state4 = np.mean(np.absolute(reversed_data[indexof_lpf4 + 1000:indexof_lpf4 + 1500,10:18] - baseline),axis = 0)
    seg_state5 = np.mean(np.absolute(reversed_data[indexof_lpf5 - 250:indexof_lpf5 + 250,10:18] - baseline),axis = 0)

    SEG_STATE4.append(seg_state4)
    SEG_STATE5.append(seg_state5)

# Step 5
# SEG_STATE4 and SEG_STATE5 are 10 * 8 matrix

init_W = np.random.rand(len(SEG_STATE4),4)
init_H = np.random.rand(4, len(SEG_STATE4[0]))

W,H = multiplication_update(SEG_STATE4, 4, thresh = 0.01,num_iter = 100,init_W = init_W, init_H = init_H,print_enabled = False)

# plot the basis vectors
# print(W)

N = 8
basis_vec1 = H[0]
basis_vec2 = H[1]
basis_vec3 = H[2]
basis_vec4 = H[3]

ind = np.arange(N) 
width = 0.2      
plt.bar(ind, basis_vec1, width, label='basis_vec1')
plt.bar(ind + width, basis_vec2, width,label='basis_vec2')
plt.bar(ind + 2*width, basis_vec3, width, label='basis_vec3')
plt.bar(ind + 3*width, basis_vec4, width,label='basis_vec4')

plt.ylabel('Activation Strength')
plt.title('Muscle synergy')

plt.xticks(ind + width / 2, ('EMG1', 'EMG2','EMG3','EMG4','EMG5','EMG6','EMG7','EMG8'))
plt.legend(loc='best')
plt.show()

