import matplotlib.pyplot as plt
import numpy as np
from process_example_data import first_last_index, multiplication_update
from plot_multiple import load_data

# Step1: Load baseline EMG data, select datas when state = 1 and compute the mean for each EMG signal
# Step2: Extract data for state 4 and 5 from MatchingTask:
#        For state 4: For the last piece of data whose state is 4, take 500 pieces of data 1 second ago
#        For state 5: For the last piece of data whose state is 5, take 250 pieces of data 0.25 second ago and 0.25 second later
# Step3: Subtract the baseline mean from the data vector and rectify them
# Step4: Compute the mean for all data points and get 2 8x1 vector for each trail
# Step5: Normalize the amplitude
# Step6: Use NMF to factorize the matrix


# Step1 Load one set of MatchingTask EMG data, select datas when the subject stays relax ,  compute the mean for each EMG signal
'''
base_time, base_torque = load_data('/home/mushenghe/Desktop/final_project/data/Oct09/c01/Right/MatchingTask/Multi_Multi_El/set01_trial01.txt', 0, 2)
plt.plot(base_time, base_torque, label = "find_baseline")
plt.xlabel('sim_time')
plt.ylabel('elbow torque')
plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct9/baseline.png')
plt.show()

# Find that from simetime = 5229-5231 the person is at relax state
# Take 2000 datapoints from index = 26994 (time = 5229) and compute the mean for these 2000 datapoints
'''

trail_data = np.loadtxt('/home/mushenghe/Desktop/final_project/data/Oct09/c01/Right/MatchingTask/Multi_Multi_El/set01_trial01.txt')
baseline = np.mean(trail_data[26994:26994 + 1000,10:18], axis = 0) 
x = np.arange(8)

''' 
plot the baseline data:

plt.bar(x, baseline)
plt.title('baseline muscle activation')
plt.xticks(x,('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct9/baseline_emg.png')
plt.show()
'''





# Step2 Find the maximum for each EMGs in each sets and subtract the baseline from it
max_set = []
max_set3_data = np.loadtxt('/home/mushenghe/Desktop/final_project/data/Oct09/c01/Right/MaxMeasurements/set03_trial01.txt')
max_set3 = np.mean(max_set3_data[6689:6689 + 1000,10:18], axis = 0) 
max_set.append(max_set3)

max_set4_data1 = np.loadtxt('/home/mushenghe/Desktop/final_project/data/Oct09/c01/Right/MaxMeasurements/set04_trial01.txt')
max_set4_data2 = np.loadtxt('/home/mushenghe/Desktop/final_project/data/Oct09/c01/Right/MaxMeasurements/set04_trial02.txt')
max_set4_1 = np.mean(max_set4_data1[2289:2289 + 1000,10:18], axis = 0) 
max_set4_2 = np.mean(max_set4_data2[22788:22788 + 1000,10:18], axis = 0) 
max_set.append(max_set4_1)
max_set.append(max_set4_2)

max_set5_data = np.loadtxt('/home/mushenghe/Desktop/final_project/data/Oct09/c01/Right/MaxMeasurements/set05_trial04.txt')
max_set5 = np.mean(max_set5_data[34065:34065 + 1000,10:18], axis = 0) 
max_set.append(max_set5)

max_set = np.absolute(np.amax(np.array(max_set),axis=0)-baseline)

'''
plot the max_set and the comparasion of max and baseline:

x = np.arange(8)
plt.bar(x, max_set)
plt.title('max_set muscle activation')
plt.xticks(x,('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct9/maxset_emg.png')
plt.show()

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

# Step 2 Extract data for state 4 and 5 from MatchingTask:

SET_TRAILS = ['set01_trial01.txt','set01_trial02.txt','set01_trial03.txt','set01_trial04.txt','set01_trial05.txt', \
    'set02_trial01.txt', 'set02_trial02.txt', 'set02_trial03.txt' ,'set03_trial01.txt', 'set03_trial02.txt']
SEG_STATE4 = []
SEG_STATE5 = []

for trail_index in range (0,10):
    trail_data = np.loadtxt('/home/mushenghe/Desktop/final_project/data/Oct09/c01/Right/MatchingTask/Multi_Multi_El/' + SET_TRAILS[trail_index])
    rows,columns = trail_data.shape
    # reverse the array to find the last data piece
    reversed_data = np.flip(trail_data,0) 
    reversed_state = reversed_data[:,1]  
    indexof_lpl4 = np.nonzero(reversed_state == 4.00000)[0]
    indexof_lpl5 = np.nonzero(reversed_state == 5.00000)[0]
    '''
    print("trail index is")
    print(trail_index)
    print("indexof_lpl4 is: ")
    print(indexof_lpl4)
    print("indexof_lpl5 is: ")
    print(indexof_lpl5)

    # found that trail5(set01_trial06), trail9(set02_trial04), trail12(set03_trial03) are empty. Delete these trails
    '''
    # index of the last piece of data whose state is 4/5 in the reversed array
    indexof_lpf4 = first_last_index(indexof_lpl4)
    indexof_lpf5 = first_last_index(indexof_lpl5)

    # Step3 Subtract the baseline data and rectify them

    seg_state4_sbase = reversed_data[indexof_lpf4 + 1000:indexof_lpf4 + 1500,10:18] - baseline
    seg_state5_sbase = reversed_data[indexof_lpf5 - 250:indexof_lpf5 + 250,10:18] - baseline

    rec_seg_state4 = np.absolute(seg_state4_sbase)
    rec_seg_state5 = np.absolute(seg_state5_sbase)

    # Step4 Take the mean

    seg_state4 = np.mean(rec_seg_state4,axis = 0)
    seg_state5 = np.mean(rec_seg_state5,axis = 0)
    
    # Step5 Normalize the amplitude
    
    seg_state4_norm = np.divide(seg_state4,max_set)
    seg_state5_norm = np.divide(seg_state5,max_set)

    SEG_STATE4.append(seg_state4_norm)
    SEG_STATE5.append(seg_state5_norm)
    
    '''
    plot one set:
    x = np.arange(8)
    width = 0.2     
    plt.bar(x, seg_state4_sbase[250], width, label='subtract_base')
    plt.bar(x + width, rec_seg_state4[250], width,label='after rectify')
    plt.bar(x + 2*width, seg_state4, width, label='mean of the state4')
    # plt.bar(x + 3*width, seg_state4_norm, width,label='after normalization')
    plt.ylabel('Muscle activation')
    plt.title('Muscle Activation for segment state4')

    plt.xticks(x + width / 2, ('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
    plt.legend(loc='best')
    # plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct9/one_Set.png')
    plt.show()
    '''


# x = np.arange(8)
# width = 0.2     
# plt.bar(x, SEG_STATE4[0], width, label='set01_trail01')
# plt.bar(x + width, SEG_STATE4[1], width,label='set01_trial02')
# plt.bar(x + 2*width, SEG_STATE4[5], width, label='set02_trial01')
# plt.bar(x + 3*width, SEG_STATE4[8], width,label='set03_trial01')
# plt.ylabel('Muscle activation')
# plt.title('Muscle Activation for segment state4')

# plt.xticks(x + width / 2, ('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
# plt.legend(loc='best')
# plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct9/4set_afternorm_state4.png')
# plt.show()

# Step 6 Use NMF to factorize the matrix
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
