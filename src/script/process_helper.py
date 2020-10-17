import numpy as np

'''
helper functions for signal processing
'''

def first_last_index(arr):
    # find the first index of the last set
    prev = arr[0]
    first = arr[-1]
    for num in arr[1:]:
        if (prev + 1) != num:
            first = prev
            break
        prev += 1
    return first

def compute_baseline_mean(path):
    # Load baseline EMG data, select datas when state = 1 and compute the mean for each EMG signal
    baselineData = np.loadtxt(path)
    target_baseline_list = baselineData[np.where(baselineData[:,1] < 1.9), 10:18]
    target_baseline = target_baseline_list.reshape((len(target_baseline_list[0]), 8))
    baseline = np.mean(target_baseline, axis = 0)
    return baseline

def standard_process(data_vec, baseline, max_vec = np.zeros((1,8))):
    # 1. Subtract the baseline vector from the data vector
    # 2. Rectify it and compute the mean for all data points
    # 3. Normalize the amplitude

    subtractbase_data_vec = np.array(data_vec) - np.array(baseline)
    target_base_data = np.mean(np.absolute(subtractbase_data_vec),axis = 0)
    if max_vec.any():
        target_base_data = np.divide(target_base_data,max_vec)
    return target_base_data    

def process_state4_5(path,SET_TRAILS,count,baseline,max_vec = np.zeros((1,8))):
    # Step 2 Extract data for state 4 and 5 from MatchingTask:

    SEG_STATE4 = []
    SEG_STATE5 = []
    for trail_index in range (0,count):
        trail_data = np.loadtxt(path + SET_TRAILS[trail_index])
        rows,columns = trail_data.shape
        # reverse the array to find the last data piece
        reversed_data = np.flip(trail_data,0) 
        reversed_state = reversed_data[:,1]  
        indexof_lpl4 = np.nonzero(reversed_state == 4.00000)[0]
        indexof_lpl5 = np.nonzero(reversed_state == 5.00000)[0]

        if (indexof_lpl4.size != 0 and indexof_lpl5.size != 0):
            # index of the last piece of data whose state is 4/5 in the reversed array
            indexof_lpf4 = first_last_index(indexof_lpl4)
            indexof_lpf5 = first_last_index(indexof_lpl5)

            state4_vec = reversed_data[indexof_lpf4 + 1000:indexof_lpf4 + 1500,10:18]
            state5_vec = reversed_data[indexof_lpf5 - 250:indexof_lpf5 + 250,10:18]

            seg_state4 = standard_process(state4_vec,baseline,max_vec)
            seg_state5 = standard_process(state5_vec,baseline,max_vec)

            SEG_STATE4.append(seg_state4)
            SEG_STATE5.append(seg_state5)

        else:
            pass
        
    return SEG_STATE4,SEG_STATE5

def 
