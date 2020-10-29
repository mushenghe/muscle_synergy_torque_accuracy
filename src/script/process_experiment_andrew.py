from process_helper import load_data,first_last_index,compute_baseline_mean,standard_process,process_state4_5,find_max_interval
from matrix_factorization import multiplication_update
from plot_multiple import plot_baseline,basisvec_N_plot
import numpy as np
import matplotlib.pyplot as plt

# Step1: Load baseline EMG data, select datas when state = 1 and compute the mean for each EMG signal
# Step2: Find the maximum for each EMGs in each sets
# Step3: Extract data for state 4 and 5 from MatchingTask:
#        For state 4: For the last piece of data whose state is 4, take 500 pieces of data 1 second ago
#        For state 5: For the last piece of data whose state is 5, take 250 pieces of data 0.25 second ago and 0.25 second later
# Step4: Subtract the baseline mean from the data vector and rectify them
# Step5: Compute the mean for all data points and get 2 8x1 vector for each trail
# Step6: Normalize the amplitude
# Step7: Use NMF to factorize the matrix




if __name__ == "__main__":

    # Step1: Find the baseline vector:
    baseline = compute_baseline_mean('/home/mushenghe/Desktop/final_project/data/Oct12/BaselineEMG/set01_trial01.txt')
    
    # Step2: Find the maximum vector:
    # use moving average:
    max_set = []
    max_path = '/home/mushenghe/Desktop/final_project/data/Oct12/MaxMeasurements/'
    SET3_TRAILS = ['set03_trial01.txt','set03_trial02.txt','set03_trial03.txt']
    SET4_TRAILS = ['set04_trial02.txt','set04_trial03.txt','set04_trial04.txt']
    SET5_TRAILS = ['set05_trial02.txt','set05_trial03.txt','set05_trial04.txt']
    max_set_trail_3, max_index_3 = find_max_interval(max_path, SET3_TRAILS, 2, 2500, 500)
    max_set_trail_4, max_index_4 = find_max_interval(max_path, SET4_TRAILS, 2, 2500, 500)
    max_set_trail_5, max_index_5 = find_max_interval(max_path, SET5_TRAILS, 2, 2500, 500)
    max_set3_data = np.loadtxt(max_path + max_set_trail_3)
    max_set5_data = np.loadtxt(max_path + max_set_trail_5)
    max_set3 = standard_process(max_set3_data[max_index_3:max_index_3 + 2500,10:18], baseline)
    max_set5 = standard_process(max_set5_data[max_index_5:max_index_5 + 2500,10:18], baseline)
    max_set.append(max_set3)
    max_set.append(max_set5)
    max_set = np.max(max_set,axis = 0)

    '''

    x = np.arange(8)
    width = 0.2     
    plt.bar(x, baseline, width, label='baseline')
    plt.bar(x + width, max_set, width,label='max_set')
    plt.ylabel('Muscle activation')
    plt.title('Muscle Activation')

    plt.xticks(x + width / 2, ('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
    plt.legend(loc='best')
    plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct9/baseline_max_compare.png')
    plt.show()

   # plot the max_set and the comparasion of max and baseline:

    x = np.arange(8)
    plt.bar(x, max_set)
    plt.title('max_set muscle activation')
    plt.xticks(x,('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
    plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct9/maxset_emg.png')
    plt.show()
   

    x = np.arange(8)
    width = 0.2     
    plt.bar(x, baseline, width, label='baseline')
    plt.bar(x + width, max_set, width,label='max_set')
    plt.ylabel('Muscle activation')
    plt.title('Muscle Activation')

    plt.xticks(x + width / 2, ('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
    plt.legend(loc='best')
    plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct9/baseline_max_compare.png')
    plt.show()
    '''


    # Step 3: Extract data for state 4 and 5 from MatchingTask:
    SET_TRAILS = []
    SET1_TRAILS = ['set01_trial01.txt','set01_trial02.txt','set01_trial03.txt','set01_trial04.txt', \
        'set01_trial05.txt', 'set01_trial06.txt', 'set01_trial07.txt']
    SET2_TRAILS = ['set02_trial01.txt','set02_trial02.txt','set02_trial03.txt','set02_trial04.txt', \
        'set02_trial05.txt', 'set02_trial06.txt', 'set02_trial07.txt', 'set02_trial08.txt', 'set02_trial09.txt']
    SET3_TRAILS = ['set03_trial01.txt','set03_trial02.txt','set03_trial03.txt','set03_trial04.txt', \
        'set03_trial05.txt', 'set03_trial06.txt', 'set03_trial07.txt', 'set03_trial08.txt', 'set03_trial09.txt']

    # SET_TRAILS.append(SET1_TRAILS)
    # SET_TRAILS.append(SET2_TRAILS)
    # SET_TRAILS.append(SET3_TRAILS)
    # Step 4,5,6: signal processing
    matching_path = '/home/mushenghe/Desktop/final_project/data/Oct12/MatchingTask/Multi_Multi_El/'
    # SEG_STATE4 = []
    # SEG_STATE5 = []
    # for i in range (3):
    #     seg_state4,seg_state5 = process_state4_5(matching_path,SET_TRAILS[i],10,baseline,max_set)
    #     SEG_STATE4.append(seg_state4)
    #     SEG_STATE5.append(seg_state5)
 
    seg_state4,seg_state5 = process_state4_5(matching_path,SET3_TRAILS,9,baseline,max_set)


    # Step 7: Use NMF to factorize the matrix
    # SEG_STATE4 and SEG_STATE5 are 10 * 8 matrix

    

    # SEG_STATE4.extend(SEG_STATE5)
    group = 4
    init_H = np.random.rand(group, 8)
    init_W = np.random.rand(len(seg_state4),group)

    W,H = multiplication_update(seg_state4, group, thresh = 0.01,num_iter = 100,init_W = init_W, init_H = init_H,print_enabled = False)
    
    # plot the basis vectors
    print(W)
    EMGs = 8
    width = 0.5  
    basisvec_N_plot(EMGs,group,H,width)
   