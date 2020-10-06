import matplotlib.pyplot as plt
import numpy as np


# load the column first column (simTime)
simTime = np.loadtxt("right_matchingtask_set01_trial09.txt")[:, 0]
EMGs = []
# load each EMG channel (column 11-18) 
with open('right_matchingtask_set01_trial09.txt','r') as f:
	for i in range (8):
		EMG = np.loadtxt("right_matchingtask_set01_trial09.txt")[:, i+10]
		EMGs.append(EMG)



# line 1 points
x = simTime
# for i in range (8):
# 		plt.plot(x, EMGs[i], label = "EMG" i)
plt.plot(x, EMGs[0], label = "EMG1")
plt.plot(x, EMGs[1], label = "EMG2")
plt.plot(x, EMGs[2], label = "EMG3")
plt.plot(x, EMGs[3], label = "EMG4")
plt.plot(x, EMGs[4], label = "EMG5")
plt.plot(x, EMGs[5], label = "EMG6")
plt.plot(x, EMGs[6], label = "EMG7")
plt.plot(x, EMGs[7], label = "EMG8")


plt.xlabel('x - axis')
# plt.xlim(560, 562)
# Set the y axis label of the current axis.
plt.ylabel('y - axis')
# Set a title of the current axes.
plt.title('EMG signal plot - EMG1')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()