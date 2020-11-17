from process_helper import load_data,first_last_index,norm_vec,compute_baseline_mean,standard_process,process_state4_5,find_max_interval
from matrix_factorization import multiplication_update,VAF
from plot_multiple import plot_baseline,basisvec_N_plot
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import itertools
# from tqdm import tqdm
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

def rank_determine_helper(A, rank, repeat_num):
    '''
    mean(global VAF)>90% & mean(local VAF) > 80%

    Choose the H corresponding to the highest global VAF:
    The synergy set corresponding to maximum VAF was considered the representative set for a given number of synergies.

    '''
    GLOBAL_VAF = []
    local_vaf = []
    train_err = []
    test_err = []
    VAF_max = 0
    H_max = []
    W_max = 0

    for repeat in range(repeat_num):
        W, H, train, test = crossval_nmf(np.array(A), rank)
        train_err.append(train[-1])
        test_err.append(test[-1])
        global_VAF,local_VAF = VAF(W,H,A)
        if global_VAF > VAF_max:
            VAF_max = global_VAF
            H_max = H
            W_max = W

        GLOBAL_VAF.append(global_VAF) #(100,)
        local_vaf.append(local_VAF) #(100,8)
        VAF_mean = np.mean(np.array(GLOBAL_VAF))

    # modify

    # print("VAF_max is: ",VAF_max)
    # print("VAF_mean is: ",VAF_mean)
    
    if VAF_mean > 90 and np.all(np.mean(local_VAF,axis = 0)> 80):
        return VAF_mean, VAF_max, H_max, W_max, train_err, test_err
    else:
        return False

    return VAF_mean, VAF_max, H_max, W_max, train_err, test_err

    
if __name__ == "__main__":

    DATA_PATH = '/home/mushenghe/Desktop/final_project/data/Oct23/'

    # Step 1: Find two baseline vectors, one for sitting one for standing
    baseline1_sit = compute_baseline_mean(DATA_PATH + 'BaselineEMG_sitting/set01_trial01.txt')
    baseline2_sit = compute_baseline_mean(DATA_PATH + 'BaselineEMG_sitting/set01_trial01.txt')
    baseline_sitting = np.mean(np.array([baseline1_sit, baseline2_sit]), axis = 0)

    baseline1_sta = compute_baseline_mean(DATA_PATH + 'BaselineEMG/set01_trial01.txt')
    baseline2_sta = compute_baseline_mean(DATA_PATH + 'BaselineEMG/set01_trial01.txt')
    baseline_standing = np.mean(np.array([baseline1_sta, baseline2_sta]), axis = 0)

    # Step 2: Use moving average to find the maximum for each EMGs:
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
    SEG_STATE4, SEG_STATE5 = np.empty((30, 8)), np.empty((30, 8))
    SEG_STATE4[:], SEG_STATE5[:] = np.nan, np.nan

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

    # Step 4: Subtract the baseline mean from the data vector and rectify them
    # Bootstrap data

    bootstrap = 20

    # # concatenate all sets together

    # SEG_STATE4, SEG_STATE5 = np.empty((600, 8)), np.empty((600, 8))
    # SEG_STATE4[:], SEG_STATE5[:] = np.nan, np.nan

    # for i in range(3):
    #     seg_state4,seg_state5 = process_state4_5(matching_path, SET_TRAILS[i], baseline_sitting, bootstrap)
    #     norm_seg4 = norm_vec(seg_state4, max_set)
    #     norm_seg5 = norm_vec(seg_state5, max_set) # 200*8
    #     # append all sets of segment 4 and 5 together in SEG_STATE4 and SEG_STATE5
    #     SEG_STATE4[i * norm_seg4.shape[0] : (i + 1) * norm_seg4.shape[0], :] = norm_seg4
    #     SEG_STATE5[i * norm_seg5.shape[0] : (i + 1) * norm_seg5.shape[0], :] = norm_seg5
   
    # # modify A
    # A = SEG_STATE4[~np.isnan(SEG_STATE4).any(axis=1)]
    
    # modify

    # # only consider one set: 

    # seg_state4,seg_state5 = process_state4_5(matching_path, SET_TRAILS[0], baseline_sitting)
    # norm_seg4 = norm_vec(seg_state4, max_set)
    # norm_seg5 = norm_vec(seg_state5, max_set) 
    # A = norm_seg4[~np.isnan(norm_seg4).any(axis=1)] # 200 * 8 

    
    # Append the max H and W together for all sets
    all_H = []
    for i in range(3):
        seg_state4,seg_state5 = process_state4_5(matching_path, SET_TRAILS[i], baseline_sitting, bootstrap)
        norm_seg4 = norm_vec(seg_state4, max_set)
        norm_seg5 = norm_vec(seg_state5, max_set) 
        A = norm_seg4[~np.isnan(norm_seg4).any(axis=1)] # 200 * 8 


        VAF_mean, VAF_max, H_max, W_max, train, test = rank_determine_helper(A, 4, 100)
        magnitute_0 = LA.norm(H_max[0])
        magnitute_1 = LA.norm(H_max[1])
        magnitute_2 = LA.norm(H_max[2])
        magnitute_3 = LA.norm(H_max[3])

        all_H.append(H_max[0]/magnitute_0)
        all_H.append(H_max[1]/magnitute_1)
        all_H.append(H_max[2]/magnitute_2)
        all_H.append(H_max[3]/magnitute_3)

        W_maxT = W_max.T
        W_maxT_C0 = W_maxT[0]*magnitute_0
        W_maxT_C1 = W_maxT[1]*magnitute_1
        W_maxT_C2 = W_maxT[2]*magnitute_2
        W_maxT_C3 = W_maxT[3]*magnitute_3

        W = np.vstack((W_maxT_C0,W_maxT_C1,W_maxT_C2,W_maxT_C3))
        print('W for set: ',i)
        print(W.T)
   

    # # modify

    # VAF_mean_last = 0
    # VAF_max_last = 0
    # H_max_last = 0
    # W_max_last = 0
    # num = 0
    
    # for rank in range(4,1,-1):
    #   print('rank is: ',rank)
    #   if rank_determine_helper(A, rank, 100):
    #       VAF_mean, VAF_max, H_max, W_max, train_err, test_err = rank_determine_helper(A, rank, 100)
    #       print("# basis vector is determined to be: ", rank)
    #   else:
    #       continue

    #   if VAF_mean >= VAF_mean_last or (VAF_mean_last - VAF_mean) < 3:
    #       VAF_mean_last, vaf_max_last, H_max_last, W_max_last = VAF_mean, VAF_max, H_max, W_max
    #       num = rank
        
    #   else:
    #       break
    
    
    
    EMGs = 8
    width = 0.5  
    ind = np.arange(EMGs) 

    plt.title('Muscle synergy for set1')   
    for i in range(1,5):
        plt.subplot(2,2,i)
        plt.bar(ind, all_H[i-1], width,label='basis_vec '+ str(i))
        plt.ylabel('Normalized Activation Strength for basis vector' + str(i) + ' of set 1')
        plt.xticks(rotation=45, ha='right')
        plt.xticks(ind, ('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
        plt.legend(loc='best')
    plt.title('Normalized Muscle synergy for set 1')

    plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct23/bootstrap_#2_set1.png')
    plt.show()

    for i in range(5,9):
        plt.subplot(2,2,i-4)
        plt.bar(ind, all_H[i-1], width,label='basis_vec '+ str(i-4))
        plt.ylabel('Normalized Activation Strength for basis vector' + str(i-4) + ' of set 2')
        plt.xticks(rotation=45, ha='right')
        plt.xticks(ind, ('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
        plt.legend(loc='best')
    plt.title('Normalized Muscle synergy for set 2')

    plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct23/bootstrap_#2_set2.png')
    plt.show()

    for i in range(9,13):
        plt.subplot(2,2,i-8)
        plt.bar(ind, all_H[i-1], width,label='basis_vec '+ str(i-8))
        plt.ylabel('Normalized Activation Strength for basis vector' + str(i-8) + ' of set 3')
        plt.xticks(rotation=45, ha='right')
        plt.xticks(ind, ('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
        plt.legend(loc='best')
    plt.title('Normalized Muscle synergy for set 3')

    plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct23/bootstrap_#2_set3.png')
    plt.show()




    

    '''
    # boxplot the RMSE of training set and validation set:

    train_error = []
    test_error = []
    for rank in range(2,5):
      VAF_mean, VAF_max, H_max, W_max, train, test = rank_determine_helper(A, rank, 100)
      train_error.append(train)
      test_error.append(test)

    ranks = np.array([2,3,4])
    train_err = np.array(train_error)
    test_err = np.array(test_error)
  

    data_a = train_error
    data_b = test_error

    ticks = ['2', '3', '4']
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure()

    bpl = plt.boxplot(data_a, positions=np.array(xrange(len(data_a)))*2.0-0.4, sym='', widths=0.6)
    bpr = plt.boxplot(data_b, positions=np.array(xrange(len(data_b)))*2.0+0.4, sym='', widths=0.6)
    set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#2C7BB6')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label='training_error')
    plt.plot([], c='#2C7BB6', label='testing_error')
    plt.legend()

    plt.xticks(xrange(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    plt.ylim(0, 8)
    plt.ylabel('RMSE')
    plt.xlabel('# of muscle synergy')
    plt.tight_layout()
    plt.savefig('nmf_compare.png')
    plt.show()
    
    '''





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