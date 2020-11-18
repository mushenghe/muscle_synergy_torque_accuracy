import numpy as np
import pandas as pd
from helper.plot_multiple import *
import matplotlib.pyplot as plt
from scipy import stats



def find_correlation(matching_data, refer_data, sunr, emg_name):
    Emg_Diff = []
    Eltorque_Diff = []
    
    Task_name = ['10%','30%','50%']
    for task in Task_name:
        emg_match = matching_data.loc[(matching_data['Su'] == sunr) & (matching_data['Task'] == task), emg_name]
        emg_ref = refer_data.loc[(refer_data['Su'] == sunr) & (refer_data['Task'] == task), emg_name]
        emg_diff = emg_match - emg_ref
        emg_diff_list = emg_diff.values.tolist()
        eltorque_diff = matching_data.loc[(matching_data['Su'] == sunr) & (matching_data['Task'] == task),'ShTorque'] \
            - refer_data.loc[(refer_data['Su'] == sunr) & (refer_data['Task'] == task),'ShTorque']
        eltorque_diff_set = eltorque_diff.values.tolist()
        Emg_Diff.append(emg_diff_list)
        Eltorque_Diff.append(eltorque_diff_set)
    
    torque_diff_list = eltorque_diff.values.tolist()
    emg_diff_list = emg_diff.values.tolist()

    return Eltorque_Diff, Emg_Diff

def plot_correlation(torque_diff_list, emg_diff_list, emg_name):
    emg_num = 8
    emg_name = ["Bicep", "Tricep", "AntDel", "MedDel",
                "PosDel", "Pec", "LowerTrap", "MidTrap"]
    colors = ['b','g','r']
    labels = ['set1','set2','set3']

    # x_all = np.empty((0,10),float)
    # y_all = np.empty((0,10),float)
    # for emg in range(emg_num):
    #     for r in range(3):
    #         x = np.array(torque_diff_list[r])
    #         y = np.array([rows[emg] for rows in emg_diff_list[r]])
    #         plt.scatter(x,y,c = colors[r],label = labels[r])
    #         print(x)
    #         x_all = np.append(x_all,x)
    #         y_all = np.append(y_all,y)

    #     print('x_all is:',x_all)
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(x_all,y_all)
    #     corr_coef = r_value * r_value

    #     plt.xlabel('Perceutual Error in Shoulder Torque')

    #     y_label = 'Matching difference in ' + emg_name[emg]
    #     plt.ylabel(y_label)

    #     plt.plot(x_all, slope*x_all + intercept)
    #     plt.title('correlation coefficient (r_squared) is : '+ str(corr_coef))
    #     plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/c07/'+ y_label +'.png')

    #     plt.show()

    x_all = np.empty((0,10),float)
    y_all = np.empty((0,10),float)

    for r in range(3):
        x = np.array(torque_diff_list[r])
        y = np.array(emg_diff_list[r]).sum(axis = 1)

        plt.scatter(x,y,c = colors[r],label = labels[r])

        x_all = np.append(x_all,x)
        y_all = np.append(y_all,y)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    corr_coef = r_value * r_value

    plt.xlabel('Perceutual Error in Shoulder Torque')

    y_label = 'Matching difference sum for all EMGs'
    plt.ylabel(y_label)

    plt.plot(x, slope*x + intercept)
    plt.title('correlation coefficient (r_squared) is : '+ str(corr_coef))
    plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/c07/'+ y_label +'.png')

    plt.show()
    



        


if __name__ == "__main__":
    path = '/home/mushenghe/Desktop/final_project/muscle_synergy/src/data/'
    emg_name = ["Bicep", "Tricep", "AntDel", "MedDel",
                "PosDel", "Pec", "LowerTrap", "MidTrap"]

    demo_data = pd.read_csv(path + 'demoInfo_111120.csv') 
    matching_data = pd.read_csv(path + 'matchData_5.csv') 
    refer_data = pd.read_csv(path + 'referenceData_5.csv') 

    torque_diff_list, emg_diff_list = find_correlation(matching_data,refer_data,'c05',emg_name)
    plot_correlation(torque_diff_list, emg_diff_list, emg_name)
    # print(np.shape(emg_diff_list))
    # print(np.shape(torque_diff_list))
