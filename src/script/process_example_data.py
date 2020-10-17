from process_helper import first_last_index,compute_baseline_mean,standard_process,process_state4_5
from matrix_factorization import multiplication_update
import numpy as np

# Step1: Load baseline EMG data, select datas when state = 1 and compute the mean for each EMG signal
# Step2: Extract data for state 4 and 5 from MatchingTask:
#        For state 4: For the last piece of data whose state is 4, take 500 pieces of data 1 second ago
#        For state 5: For the last piece of data whose state is 5, take 250 pieces of data 0.25 second ago and 0.25 second later
# Step3: Subtract the baseline vector from the data vector
# Step4: Rectify them and compute the mean for all data points, get 2 8x1 vector for each trail
# Step5: Use NMF to factorize the matrix


if __name__ == "__main__":

    # Step1: Load baseline EMG data, select datas when state = 1 and compute the mean for each EMG signal
    SET1_TRAILS = ['set01_trial01.txt','set01_trial02.txt','set01_trial03.txt','set01_trial04.txt','set01_trial05.txt','set01_trial06.txt', \
        'set01_trial07.txt', 'set01_trial08.txt', 'set01_trial09.txt' ,'set01_trial10.txt']
    baseline = compute_baseline_mean('/home/mushenghe/Desktop/final_project/data/BaselineEMG/set01_trial01.txt')

    # Step 2,3,4: Extract data for state 4 and 5 from MatchingTask:
    SEG_STATE4,SEG_STATE5 = process_state4_5('/home/mushenghe/Desktop/final_project/data/MatchingTask/',SET1_TRAILS,10,baseline)
    # Step 5
    # SEG_STATE4 and SEG_STATE5 are 10 * 8 matrix
    init_W = np.random.rand(len(SEG_STATE4),4)
    init_H = np.random.rand(4, len(SEG_STATE4[0]))

    W,H = multiplication_update(SEG_STATE4, 4, thresh = 0.01,num_iter = 100,init_W = init_W, init_H = init_H,print_enabled = False)

    N = 8
    width = 0.2   
    basisvec_one_plot(N,H,width)
