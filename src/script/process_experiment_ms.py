from process_helper import load_data,first_last_index,compute_baseline_mean,standard_process,process_state4_5,find_max_interval
from matrix_factorization import multiplication_update
from plot_multiple import load_data,plot_baseline,basisvec_N_plot
import numpy as np

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
    '''
    # Load one set of MatchingTask EMG data, select datas when the subject stays relax, compute the mean for each EMG signal
    base_time, base_torque = load_data('/home/mushenghe/Desktop/final_project/data/Oct09/c01/Right/MatchingTask/Multi_Multi_El/set01_trial01.txt', 0, 2)
    plt.plot(base_time, base_torque, label = "find_baseline")
    plt.xlabel('sim_time')
    plt.ylabel('elbow torque')
    plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct9/baseline.png')
    plt.show()
    '''
    # Find that from simetime = 5229-5231 the person is at relax state
    # Take 2000 datapoints from index = 26994 (time = 5229) and compute the mean for these 2000 datapoints

    trail_data = np.loadtxt('/home/mushenghe/Desktop/final_project/data/Oct09/c01/Right/MatchingTask/Multi_Multi_El/set01_trial01.txt')
    baseline = np.mean(trail_data[26994:26994 + 1000,10:18], axis = 0) 
    # plot_baseline(baseline,8)     //plot the baseline data

    # Step2: Find the maximum vector:

    ''' 
    use moving average:

    SET3_TRAILS = ['set03_trial01.txt','set03_trial02.txt','set03_trial03.txt','set03_trial04.txt']
    max_set_trail, max_index = find_max_interval('/home/mushenghe/Desktop/final_project/data/Oct09/c01/Right/MaxMeasurements/', SET3_TRAILS, 2, 2500, 500)
    '''

    max_set = []
    max_set3_data = np.loadtxt('/home/mushenghe/Desktop/final_project/data/Oct09/c01/Right/MaxMeasurements/set03_trial01.txt')

    max_set3 = standard_process(max_set3_data[6689:6689 + 1000,10:18], baseline)
    max_set.append(max_set3)

    max_set4_data1 = np.loadtxt('/home/mushenghe/Desktop/final_project/data/Oct09/c01/Right/MaxMeasurements/set04_trial01.txt')
    max_set4_data2 = np.loadtxt('/home/mushenghe/Desktop/final_project/data/Oct09/c01/Right/MaxMeasurements/set04_trial02.txt')
    max_set4_1 = standard_process(max_set4_data1[2289:2289 + 1000,10:18], baseline)
    max_set4_2 = standard_process(max_set4_data2[22788:22788 + 1000,10:18], baseline)
    max_set.append(max_set4_1)
    max_set.append(max_set4_2)

    max_set5_data = np.loadtxt('/home/mushenghe/Desktop/final_project/data/Oct09/c01/Right/MaxMeasurements/set05_trial04.txt')
    max_set5 = standard_process(max_set5_data[34065:34065 + 1000,10:18], baseline)
    max_set.append(max_set5)

    max_set = np.max(max_set,axis = 0)

    '''
    plot the max_set and the comparasion of max and baseline:

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
    SET_TRAILS = ['set01_trial01.txt','set01_trial02.txt','set01_trial03.txt','set01_trial04.txt','set01_trial05.txt', \
        'set02_trial01.txt', 'set02_trial02.txt', 'set02_trial03.txt' ,'set03_trial01.txt', 'set03_trial02.txt']
    
    # Step 4,5,6: signal processing
    SEG_STATE4,SEG_STATE5 = process_state4_5('/home/mushenghe/Desktop/final_project/data/Oct09/c01/Right/MatchingTask/Multi_Multi_El/',SET_TRAILS,10,baseline,max_set)

    # Step 7: Use NMF to factorize the matrix
    # SEG_STATE4 and SEG_STATE5 are 10 * 8 matrix

    group = 4
    init_H = np.random.rand(group, 8)

    # SEG_STATE4.extend(SEG_STATE5)
    init_W = np.random.rand(len(SEG_STATE5),group)

    W,H = multiplication_update(SEG_STATE5, group, thresh = 0.01,num_iter = 100,init_W = init_W, init_H = init_H,print_enabled = False)
    
    # plot the basis vectors
    print(W)
    EMGs = 8
    width = 0.5  
    basisvec_N_plot(EMGs,group,H,width)