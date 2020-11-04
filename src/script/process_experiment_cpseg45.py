from process_helper import load_data,first_last_index,norm_vec,compute_baseline_mean,standard_process,process_state4_5,find_max_interval
from matrix_factorization import multiplication_update
from plot_multiple import plot_baseline,basisvec_N_plot
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from numpy.random import randn, rand
from nmf_crosscal import * 

# Step1: Load BaselineEMG_sitting data, select datas when state = 1 and compute the mean for each EMG signal
# Step2: Find the maximum for each EMGs
# Step3: Extract data for state 4 and 5 from MatchingTask:
#        For state 4: For the last piece of data whose state is 4, take 500 pieces of data 1 second ago
#        For state 5: For the last piece of data whose state is 5, take 250 pieces of data 0.25 second ago and 0.25 second later
# Step4: Subtract the baseline mean from the data vector and rectify them
# Step5: Compute the mean for all data points and get 2 8x1 vector for each trail
# Step6: Normalize the amplitude
# Step7: Use NMF to factorize the matrix
'''
parameters:

baseline_sitting - baseline when sitting, used for all tasks except for the maxMeasurements
baseline_standing - baseline when standing
max_bicep - max value for Bicep
max_tricep - max value for Tricep lateral
max_andeltoid - max value for Anterior deltoid
max_medeltoid - max value for Medial deltoid
max_posdeltoid - max value for Posterior deltoid
max_lotrap - max value for Pectoralis major
max_pec - max value for Lower trapezius
max_midtrap - max value for Middle trapezius

'''


if __name__ == "__main__":

    DATA_PATH = '/home/mushenghe/Desktop/final_project/data/Oct23/' 

    # Step1: Find two baseline vectors, one for sitting one for standing
    baseline1_sit = compute_baseline_mean(DATA_PATH + 'BaselineEMG_sitting/set01_trial01.txt')
    baseline2_sit = compute_baseline_mean(DATA_PATH + 'BaselineEMG_sitting/set01_trial01.txt')
    baseline_sitting = np.mean(np.array([baseline1_sit, baseline2_sit]), axis = 0)

    baseline1_sta = compute_baseline_mean(DATA_PATH + 'BaselineEMG/set01_trial01.txt')
    baseline2_sta = compute_baseline_mean(DATA_PATH + 'BaselineEMG/set01_trial01.txt')
    baseline_standing = np.mean(np.array([baseline1_sta, baseline2_sta]), axis = 0)

    # Step2: Use moving average to find the maximum for each EMGs:
    # Note that the baseline for bicep and tricep is baseline_sitting andthat for other muscles is baseline_standing

    MAX_TRAILS = []

    max_set = []
    

    BI_MAX_TRAILS = ['set06_trial01.txt','set06_trial02.txt','set06_trial03.txt','set06_trial04.txt']
    MAX_TRAILS.append(BI_MAX_TRAILS)

    TRI_MAX_TRAILS = ['set07_trial01.txt','set07_trial02.txt','set07_trial03.txt','set07_trial04.txt']
    MAX_TRAILS.append(TRI_MAX_TRAILS)

    ANDEL_MAX_TRAILS = ['set03_trial01.txt','set03_trial02.txt','set03_trial03.txt']
    MAX_TRAILS.append(ANDEL_MAX_TRAILS)

    MEDEL_MAX_TRAILS = ['set04_trial02.txt','set04_trial03.txt','set04_trial04.txt']
    MAX_TRAILS.append(MEDEL_MAX_TRAILS)

    POSDEL_MAX_TRAILS = ['set05_trial02.txt','set05_trial03.txt','set05_trial04.txt']
    MAX_TRAILS.append(POSDEL_MAX_TRAILS)

    PEC_MAX_TRAILS = ['set01_trial01.txt','set01_trial02.txt','set01_trial03.txt']
    MAX_TRAILS.append(PEC_MAX_TRAILS)

    LOTRAP_MAX_TRAILS = ['set04_trial02.txt','set04_trial03.txt','set04_trial04.txt']
    MAX_TRAILS.append(MEDEL_MAX_TRAILS)

    MIDTRAP_MAX_TRAILS = ['set02_trial02.txt','set02_trial03.txt','set02_trial04.txt']
    MAX_TRAILS.append(MIDTRAP_MAX_TRAILS)



    for i in range(2):
        # find the trail set and the index of the max moving window starting index
        # set the maximum emg value
        max_set.append(find_max_interval(DATA_PATH + 'MaxMeasurements/', MAX_TRAILS[i], 10+i, baseline_sitting[i], 1000, 5))

    for i in range(2,8):
        max_set.append(find_max_interval(DATA_PATH + 'MaxMeasurements/', MAX_TRAILS[i], 10+i, baseline_standing[i], 1000, 5))
    
    '''
    x = np.arange(8)
    width = 0.2     
    plt.bar(x, baseline_standing, width, label='baseline_standing')
    plt.bar(x + width, max_set, width,label='max_set')
    plt.ylabel('Muscle activation')
    plt.title('Muscle Activation')

    plt.xticks(x + width / 2, ('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
    plt.legend(loc='best')
    plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct23/baseline_max_compare.png')
    plt.show()
    
    '''

    # Step 3: Extract data for state 4 and 5 from MatchingTask:
    SET_TRAILS = []
    SEG_STATE4 = []
    SEG_STATE5 = []
    SEG = []

    SET1_TRAILS = ['set01_trial01.txt','set01_trial02.txt','set01_trial03.txt','set01_trial04.txt', 'set01_trial05.txt', \
        'set01_trial06.txt', 'set01_trial07.txt', 'set01_trial08.txt', 'set01_trial09.txt', 'set01_trial10.txt']
    SET_TRAILS.append(SET1_TRAILS)

    SET2_TRAILS = ['set02_trial01.txt','set02_trial02.txt','set02_trial03.txt','set02_trial04.txt', \
        'set02_trial05.txt', 'set02_trial06.txt', 'set02_trial07.txt', 'set02_trial08.txt', 'set02_trial09.txt', 'set02_trial10.txt']
    SET_TRAILS.append(SET2_TRAILS)

    SET3_TRAILS = ['set03_trial01.txt','set03_trial02.txt','set03_trial03.txt','set03_trial04.txt', \
        'set03_trial05.txt', 'set03_trial06.txt', 'set03_trial07.txt', 'set03_trial08.txt', 'set03_trial09.txt', 'set03_trial10.txt']
    SET_TRAILS.append(SET3_TRAILS)

    SET_TRAILS = np.array(SET_TRAILS).reshape(30,)

    print(np.shape(SET_TRAILS))

    matching_path = DATA_PATH + 'MatchingTask/Multi_Multi_El/'


    seg_state4,seg_state5 = process_state4_5(matching_path, SET_TRAILS, baseline_sitting)

    SEG.append(seg_state4)
    SEG.append(seg_state5)
    
    train_err = []
    test_err = []
    repeat_range = range(100)
    set_range = range(2)
    synergy_range = range(2)

    min = 100
    min_H = randn(2, 8)
    all_H = []
    norm_H = []

    for set_index in set_range:
        for repeat in repeat_range:
            W, H, train, test = crossval_nmf(np.array(SEG[set_index]), 2)
            train_err.append(train[-1])
            test_err.append(test[-1])
            if test[-1] < min:
                min = test[-1]
                min_H = H
        all_H.append(min_H[0]/LA.norm(min_H[0]))
        all_H.append(min_H[1]/LA.norm(min_H[1]))
        
    print(np.shape(all_H))
    
  
    # plot the basis vectors
    EMGs = 8
    width = 0.5  
    ind = np.arange(EMGs) 

    for i in range(1,3):
        plt.subplot(1,2,i)
        plt.bar(ind, all_H[i-1], width,label='basis_vec '+ str(i))
        plt.ylabel('Normalized Activation Strength for basis vector' + str(i) + ' of segment 4')
        plt.xticks(rotation=45, ha='right')
        plt.xticks(ind, ('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
        plt.legend(loc='best')
    plt.title('Normalized Muscle synergy for segment 4')

    plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct23/basis_vec_seg4.png')
    plt.show()

    for i in range(3,5):
        plt.subplot(1,2,i-2)
        plt.bar(ind, all_H[i-1], width,label='basis_vec '+ str(i-2))
        plt.ylabel('Normalized Activation Strength for basis vector' + str(i-2) + ' of segment 5')
        plt.xticks(rotation=45, ha='right')
        plt.xticks(ind, ('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
        plt.legend(loc='best')
    plt.title('Normalized Muscle synergy for segment 5')

    plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct23/basis_vec_seg5.png')
    plt.show()

    

    
    
    # print(np.shape(seg_state4.size))
    '''
    # Step4 take seg_state 4 and plot the RMSE for training set and validation set:

    ranks = []
    train_err = []
    test_err = []

    rank_range = range(2,5)
    repeat_range = range(100)

    with tqdm(total=len(rank_range)*len(repeat_range)) as pbar:
    for rank, repeat in itertools.product(rank_range, repeat_range):
        _, _, train, test = crossval_nmf(np.array(seg_state4), rank)
        ranks.append(rank)
        train_err.append(train[-1])
        test_err.append(test[-1])
        
    
    ranks = np.array(ranks)
    train_err = np.array(train_err)
    test_err = np.array(test_err)

    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(7, 4))
    axes[0].plot(ranks + randn(ranks.size)*.1, train_err, '.k')
    axes[0].plot(np.unique(ranks), [np.mean(train_err[ranks==r]) for r in np.unique(ranks)], '-r', zorder=-1)
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('Train Error')
    axes[0].set_xlabel('# of basis vectors')

    axes[1].plot(ranks + randn(ranks.size)*.1, test_err, '.k')
    axes[1].plot(np.unique(ranks), [np.mean(test_err[ranks==r]) for r in np.unique(ranks)], '-r', zorder=-1)
    axes[1].set_xticks(np.unique(ranks).astype(int))
    axes[1].set_title('Test Error')
    axes[1].set_xlabel('# of basis vectors')

    fig.tight_layout()
    fig.savefig('cv_nmf.jpg')
    plt.show()
    '''
    
   