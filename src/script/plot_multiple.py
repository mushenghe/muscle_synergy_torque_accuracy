import matplotlib.pyplot as plt
import numpy as np

def plot_emg_1trail(path):
    # load the column first column (simTime)
    simTime = np.loadtxt(path)[:, 0]
    EMGs = []
    # load each EMG channel (column 11-18) 
    with open(path,'r') as f:
        for i in range (8):
            EMG = np.loadtxt(path)[:, i+10]
            EMGs.append(EMG)

    # line 1 points
    x = simTime
    plt.plot(x, EMGs[0], label = "EMG1")
    plt.plot(x, EMGs[1], label = "EMG2")
    plt.plot(x, EMGs[2], label = "EMG3")
    plt.plot(x, EMGs[3], label = "EMG4")
    plt.plot(x, EMGs[4], label = "EMG5")
    plt.plot(x, EMGs[5], label = "EMG6")
    plt.plot(x, EMGs[6], label = "EMG7")
    plt.plot(x, EMGs[7], label = "EMG8")


    plt.xlabel('x - axis')
    # Set the y axis label of the current axis.
    plt.ylabel('y - axis')
    # Set a title of the current axes.
    plt.title('EMG signal plot - EMG1')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()

def plot_baseline(baseline, N):

    x = np.arange(N)
    plt.bar(x, baseline)
    plt.title('baseline muscle activation')
    plt.xticks(x,('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
    # plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct9/baseline_emg.png')
    plt.show()

def basisvec_one_plot(N,basis_vec,width):
    # plot the basis vectors
    ind = np.arange(N)    
    plt.bar(ind, basis_vec[0], width, label='basis_vec1')
    plt.bar(ind + width, basis_vec[1], width,label='basis_vec2')
    plt.bar(ind + 2*width, basis_vec[2], width, label='basis_vec3')
    plt.bar(ind + 3*width, basis_vec[3], width,label='basis_vec4')

    plt.ylabel('Activation Strength')
    plt.title('Muscle synergy')

    plt.xticks(ind + width / 2, ('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
    plt.legend(loc='best')
    plt.show()

def basisvec_N_plot(N,basis_vec,width):
    ind = np.arange(N) 

    for i in range(1,N+1):
        plt.subplot(2,N/2,i)
        plt.bar(ind + (i-1)*width, basis_vec[i-1], width,label='basis_vec'+'i')

    plt.ylabel('Activation Strength')
    plt.title('Muscle synergy')   

    # plt.bar(ind + width, basis_vec2, width,label='basis_vec2')
    # plt.bar(ind + 2*width, basis_vec3, width, label='basis_vec3')
    # plt.bar(ind + 3*width, basis_vec4, width,label='basis_vec4')

    plt.xticks(ind + width / 2, ('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
    plt.legend(loc='best')
    #plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct9/4set_afternorm_state4.png')
    plt.show()




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


'''
plot the maxmeasure data and find the 1000 datapoint of the maxmum torque for each trail

# Step1: Plot all trails and find the corresponding time period:
#   For set 03 and 04 choose 1000 datapoints where their elbow torque is the largest
#   For set05 choose 1500 datapoints where their shoulder torque is the largest

# Step2: Compute the mean for each EMG in each trail, put them in a large matrix ([all trails in all sets] * 8), 
# find the largest number in each column
'''


'''
# LOAD DATA
SET3_TRAILS = ['set03_trial01.txt','set03_trial02.txt','set03_trial03.txt','set03_trial04.txt']
SET4_TRAILS = ['set04_trial01.txt','set04_trial02.txt','set04_trial03.txt']
SET5_TRAILS = ['set05_trial01.txt','set05_trial02.txt','set05_trial03.txt','set05_trial04.txt','set05_trial05.txt','set05_trial06.txt']

X = []
Y = []

# Plot set3
for data_file in SET3_TRAILS:
    x, y = load_data('/home/mushenghe/Desktop/final_project/data/Oct09/c01/Right/MaxMeasurements/'+ data_file, 0,2)
    X.append(x)
    Y.append(y)


fig1, axs = plt.subplots(2, 2,figsize=(15, 10))

axs[0, 0].plot(X[0], Y[0])
axs[0, 0].set_title('trail01')
axs[0, 1].plot(X[1], Y[1])
axs[0, 1].set_title('trail02')
axs[1, 0].plot(X[2], Y[2])
axs[1, 0].set_title('trail03')
axs[1, 1].plot(X[3], Y[3])
axs[1, 1].set_title('trail04')

for ax in axs.flat:
    ax.set(xlabel='sim_time', ylabel='elbow torque')
fig1.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct9/Max_set3.png')
'''
'''
For set3, choose 1000 datapoints in trail 01 from index 6689, compute the mean
'''
'''
# Plot set4
for data_file in SET4_TRAILS:
    x, y = load_data('/home/mushenghe/Desktop/final_project/data/Oct09/c01/Right/MaxMeasurements/'+ data_file, 0,2)
    X.append(x)
    Y.append(y)


fig1, axs = plt.subplots(1, 3,figsize=(25, 5))

axs[0].plot(X[0], Y[0])
axs[0].set_title('trail01')
axs[1].plot(X[1], Y[1])
axs[1].set_title('trail02')
axs[2].plot(X[2], Y[2])
axs[2].set_title('trail03')

for ax in axs.flat:
    ax.set(xlabel='sim_time', ylabel='elbow torque')
fig1.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct9/Max_set4.png')
'''
'''
For set4, choose 1000 datapoints in trail 01 from 2289 and in trail 02 from 22788, compute the mean
'''
'''
for data_file in SET5_TRAILS:
    x, y = load_data('/home/mushenghe/Desktop/final_project/data/Oct09/c01/Right/MaxMeasurements/'+ data_file, 0, 3)
    X.append(x)
    Y.append(y)


fig1, axs = plt.subplots(2, 3,figsize=(25, 10))

axs[0,0].plot(X[0], Y[0])
axs[0,0].set_title('trail01')
axs[0,1].plot(X[1], Y[1])
axs[0,1].set_title('trail02')
axs[0,2].plot(X[2], Y[2])
axs[0,2].set_title('trail03')
axs[1,0].plot(X[3], Y[3])
axs[1,0].set_title('trail04')
axs[1,1].plot(X[4], Y[4])
axs[1,1].set_title('trail05')
axs[1,2].plot(X[5], Y[5])
axs[1,2].set_title('trail06')
for ax in axs.flat:
    ax.set(xlabel='sim_time', ylabel='shoulder torque')
fig1.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct9/Max_set5.png')

For set4, choose 1000 datapoints in trail 04 from index 34065, compute the mean
'''
