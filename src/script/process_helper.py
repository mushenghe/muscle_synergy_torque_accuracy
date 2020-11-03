import numpy as np

'''
helper functions for signal processing
'''
def load_data(txt_path, column_x, column_y):
    """
    Loads data from txt files 

    Args:
        txt_path (str): path to txt file containing the data (e.g. 'data/blobs.json')
    Returns:
        features (np.ndarray): numpy array containing the x values
        targets (np.ndarray): numpy array containing the y values in the range -1, 1.
    """

    x = np.loadtxt(txt_path)[:, column_x]
    y = np.loadtxt(txt_path)[:, column_y]

    return x, y

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

def norm_vec(target_vec,max_vec = np.zeros((1,8))):
    # Normalize the amplitude
    target_base_data = np.divide(target_vec,max_vec)
    return target_vec    

def standard_process(data_vec, baseline):
    # 1. Subtract the baseline vector from the data vector
    # 2. Rectify it and compute the mean for all data points

    subtractbase_data_vec = np.array(data_vec) - np.array(baseline)
    target_base_data = np.mean(np.absolute(subtractbase_data_vec),axis = 0)
    
    return target_base_data    

def process_state4_5(path,SET_TRAILS,baseline):
    # Step 2 Extract data for state 4 and 5 from MatchingTask:

    SEG_STATE4 = []
    SEG_STATE5 = []
    for trail_index in range (0,len(SET_TRAILS)):
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

            seg_state4 = standard_process(state4_vec,baseline)
            seg_state5 = standard_process(state5_vec,baseline)

            SEG_STATE4.append(seg_state4)
            SEG_STATE5.append(seg_state5)

        else:
            pass
        
    return SEG_STATE4,SEG_STATE5

def find_max_interval(path, SET_TRAILS, column_index, baseline, window_size, step):
    Max = 0
    max_index = 0
    max_set_trail = ''
    for data_file in SET_TRAILS:
        target_vec = np.absolute(np.loadtxt(path + data_file)[:, column_index] - np.array(baseline))
        for index in range(0,len(target_vec)-window_size,step):
            interval_mean = np.mean(target_vec[index : index + window_size])
            if interval_mean > Max:
                Max = interval_mean
                max_index = index
                max_set_trail = data_file

    return Max
