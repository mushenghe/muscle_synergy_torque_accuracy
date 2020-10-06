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

# Step1
baselineData = np.loadtxt("/home/mushenghe/Desktop/final_project/muscle_synergy/data/BaselineEMG/set01_trial01.txt")
target_baseline = baselineData[np.where(baselineData[:,1] < 1.9), 10:18].reshape((6401, 8))
noise = np.mean(target_baseline, axis = 0)


# Step 2,3,4
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
    seg_state4 = np.mean(reversed_data[indexof_lpf4 + 1000:indexof_lpf4 + 1500,10:18], axis = 0) - noise
    seg_state5 = np.mean(reversed_data[indexof_lpf5 - 250:indexof_lpf5 + 250,10:18], axis = 0) - noise

    SEG_STATE4.append(seg_state4)
    SEG_STATE5.append(seg_state5)

# Step 5


