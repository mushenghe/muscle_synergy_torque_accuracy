from process_helper import load_data,first_last_index,norm_vec,compute_baseline_mean,standard_process,process_state4_5,find_max_interval
from matrix_factorization import multiplication_update,VAF
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

def rank_determine_helper(A,repeat_num):
    '''
    mean(global VAF)>90% & mean(local VAF) > 80%

    Choose the H corresponding to the highest global VAF:
    The synergy set corresponding to maximum VAF was considered the representative set for a given number of synergies.

    '''
    GLOBAL_VAF = []
    local_vaf = []
    VAF_max = 0
    H_max = []
    W_max = 0

    for repeat in range(repeat_num):
        W, H = multiplication_update(A,rank)
        global_VAF,local_VAF = VAF(W,H,A)
        if global_VAF > VAF_max:
            VAF_max = global_VAF
            H_max = H
            W_max = W

        GLOBAL_VAF.append(global_VAF) #(100,)
        local_vaf.append(local_VAF) #(100,8)
        VAF_mean = np.mean(np.array(GLOBAL_VAF))
    
    if VAF_mean > 90 and np.all(np.mean(local_VAF,axis = 0)> 80):
        return VAF_mean, VAF_max, H_max, W_max
    else:
        return False


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
    # plot the maximum set

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
    SEG_STATE4 = np.array([[]])
    SEG_STATE5 = np.array([[]])

    SET1_TRAILS = ['set01_trial01.txt','set01_trial02.txt','set01_trial03.txt','set01_trial04.txt', 'set01_trial05.txt', \
        'set01_trial06.txt', 'set01_trial07.txt', 'set01_trial08.txt', 'set01_trial09.txt', 'set01_trial10.txt']
    SET_TRAILS.append(SET1_TRAILS)

    SET2_TRAILS = ['set02_trial01.txt','set02_trial02.txt','set02_trial03.txt','set02_trial04.txt', \
        'set02_trial05.txt', 'set02_trial06.txt', 'set02_trial07.txt', 'set02_trial08.txt', 'set02_trial09.txt', 'set02_trial10.txt']
    SET_TRAILS.append(SET2_TRAILS)

    SET3_TRAILS = ['set03_trial01.txt','set03_trial02.txt','set03_trial03.txt','set03_trial04.txt', \
        'set03_trial05.txt', 'set03_trial06.txt', 'set03_trial07.txt', 'set03_trial08.txt', 'set03_trial09.txt', 'set03_trial10.txt']
    SET_TRAILS.append(SET3_TRAILS)

    matching_path = DATA_PATH + 'MatchingTask/Multi_Multi_El/'
    # print(SET_TRAILS)

    # append all sets of segment 4 and 5 together in SEG_STATE4 and SEG_STATE5
    for i in range(3):
        seg_state4,seg_state5 = process_state4_5(matching_path, SET_TRAILS[i], baseline_sitting)
        norm_seg4 = norm_vec(seg_state4, max_set)
        norm_seg5 = norm_vec(seg_state4, max_set)
        SEG_STATE4 = np.append(SEG_STATE4, norm_seg4, axis = 0)
        SEG_STATE5 = np.append(SEG_STATE5, norm_seg5, axis = 0)
        

    # seg_state4,seg_state5 = process_state4_5(matching_path, SET_TRAILS[0], baseline_sitting)
    # norm_seg4 = norm_vec(seg_state4, max_set)
    
    A = SEG_STATE4
    print(np.shape(A))

    VAF_mean_last = 0
    VAF_max_last = 0
    H_max_last = 0
    W_max_last = 0
    num = 0

    for rank in range(4,1,-1):
      if rank_determine_helper(A,rank):
          VAF_mean, VAF_max, H_max, W_max = rank_determine_helper(A,rank)
          print("# basis vector is determined to be: ", rank)
          print(" VAF_mean : ",VAF_mean)
          print(" VAF is : ",VAF_max)
      else:
          continue

      if VAF_mean >= VAF_mean_last or (VAF_mean_last - VAF_mean) < 3:
          VAF_mean_last, vaf_max_last, H_max_last, W_max_last = VAF_mean, VAF_max, H_max, W_max
          num = rank
        
      else:
          break

    




        





  



    '''
    # plot the basis vectors
    EMGs = 8
    width = 0.5  
    ind = np.arange(EMGs) 

    plt.title('Muscle synergy for set1')   
    for i in range(1,3):
        plt.subplot(1,2,i)
        plt.bar(ind, all_H[i-1], width,label='basis_vec '+ str(i))
        plt.ylabel('Normalized Activation Strength for basis vector' + str(i) + ' of set 1')
        plt.xticks(rotation=45, ha='right')
        plt.xticks(ind, ('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
        plt.legend(loc='best')
    plt.title('Normalized Muscle synergy for set 1')

    plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct23/basis_vec_set1_seg4.png')
    plt.show()

    for i in range(3,5):
        plt.subplot(1,2,i-2)
        plt.bar(ind, all_H[i-1], width,label='basis_vec '+ str(i-2))
        plt.ylabel('Normalized Activation Strength for basis vector' + str(i-2) + ' of set 2')
        plt.xticks(rotation=45, ha='right')
        plt.xticks(ind, ('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
        plt.legend(loc='best')
    plt.title('Normalized Muscle synergy for set 2')

    plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct23/basis_vec_set2_seg4.png')
    plt.show()

    for i in range(5,7):
        plt.subplot(1,2,i-4)
        plt.bar(ind, all_H[i-1], width,label='basis_vec '+ str(i-4))
        plt.ylabel('Normalized Activation Strength for basis vector' + str(i-4) + ' of set 3')
        plt.xticks(rotation=45, ha='right')
        plt.xticks(ind, ('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
        plt.legend(loc='best')
    plt.title('Normalized Muscle synergy for set 3')

    plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct23/basis_vec_set3_seg4.png')
    plt.show()
    '''