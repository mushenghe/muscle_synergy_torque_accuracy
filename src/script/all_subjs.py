from helper.process_helper import *
from helper.matrix_factorization import *
from helper.plot_multiple import *
from helper.read_data import *
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import itertools
from numpy.random import randn, rand
from helper.nmf_crosscal import *

# Step1: Load the normalized reference data and matching data for all subjects, group them in sets
# Step2: Apply NMF on each sets for both reference and matching data
# Step3: Perform NMF and compute VAF for all of them


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

    print("VAF_max is: ",VAF_max)
    print("VAF_mean is: ",VAF_mean)
    
    if VAF_mean > 90 and np.all(np.mean(local_VAF,axis = 0)> 80):
        return VAF_mean, VAF_max, H_max, W_max, train_err, test_err
    else:
        return False

    return VAF_mean, VAF_max, H_max, W_max, train_err, test_err

    
if __name__ == "__main__":

    path = '/home/mushenghe/Desktop/final_project/muscle_synergy/src/data/'
    emg_name = ["Bicep", "Tricep", "AntDel", "MedDel",
                "PosDel", "Pec", "LowerTrap", "MidTrap"]

    demo_data = pd.read_csv(path + 'demoInfo_4su.csv') 
    matching_data = pd.read_csv(path + 'matchData_4su.csv') 
    refer_data = pd.read_csv(path + 'referenceData_4su.csv') 

    Emg_match, Emg_ref = select_sets(matching_data, refer_data, emg_name) #(3,40,8)

    for set_index in range (3):
        print("set number is: ",set_index)
        A = Emg_ref[set_index]
        B = Emg_match[set_index]

        VAF_mean_last = 0
        VAF_max_last = 0
        H_max_last = 0
        W_max_last = 0
        num = 0

        for rank in range(4,1,-1):
            print('rank is: ',rank)
            if rank_determine_helper(A, rank, 100):
                VAF_mean, VAF_max, H_max, W_max, train_err, test_err = rank_determine_helper(B, rank, 100)
            else:
                continue

            if VAF_mean >= VAF_mean_last or (VAF_mean_last - VAF_mean) < 3:
                VAF_mean_last, vaf_max_last, H_max_last, W_max_last = VAF_mean, VAF_max, H_max, W_max
                num = rank
                print("# basis vector for set" + str(set_index) + "is determined to be: " + str(num))
            else:
                break
        
        # A = norm_seg4[~np.isnan(norm_seg4).any(axis=1)] # 200 * 8 


        # VAF_mean, VAF_max, H_max, W_max, train, test = rank_determine_helper(A, 4, 100)
        # magnitute_0 = LA.norm(H_max[0])
        # magnitute_1 = LA.norm(H_max[1])
        # magnitute_2 = LA.norm(H_max[2])
        # magnitute_3 = LA.norm(H_max[3])

        # all_H.append(H_max[0]/magnitute_0)
        # all_H.append(H_max[1]/magnitute_1)
        # all_H.append(H_max[2]/magnitute_2)
        # all_H.append(H_max[3]/magnitute_3)

        # W_maxT = W_max.T
        # W_maxT_C0 = W_maxT[0]*magnitute_0
        # W_maxT_C1 = W_maxT[1]*magnitute_1
        # W_maxT_C2 = W_maxT[2]*magnitute_2
        # W_maxT_C3 = W_maxT[3]*magnitute_3

        # W = np.vstack((W_maxT_C0,W_maxT_C1,W_maxT_C2,W_maxT_C3))
        # print('W for set: ',i)
        # print(W.T)
   

    
    
    
    # EMGs = 8
    # width = 0.5  
    # ind = np.arange(EMGs) 

    # plt.title('Muscle synergy for set1')   
    # for i in range(1,5):
    #     plt.subplot(2,2,i)
    #     plt.bar(ind, all_H[i-1], width,label='basis_vec '+ str(i))
    #     plt.ylabel('Normalized Activation Strength for basis vector' + str(i) + ' of set 1')
    #     plt.xticks(rotation=45, ha='right')
    #     plt.xticks(ind, ('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
    #     plt.legend(loc='best')
    # plt.title('Normalized Muscle synergy for set 1')

    # plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct23/bootstrap_#2_set1.png')
    # plt.show()

    # for i in range(5,9):
    #     plt.subplot(2,2,i-4)
    #     plt.bar(ind, all_H[i-1], width,label='basis_vec '+ str(i-4))
    #     plt.ylabel('Normalized Activation Strength for basis vector' + str(i-4) + ' of set 2')
    #     plt.xticks(rotation=45, ha='right')
    #     plt.xticks(ind, ('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
    #     plt.legend(loc='best')
    # plt.title('Normalized Muscle synergy for set 2')

    # plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct23/bootstrap_#2_set2.png')
    # plt.show()

    # for i in range(9,13):
    #     plt.subplot(2,2,i-8)
    #     plt.bar(ind, all_H[i-1], width,label='basis_vec '+ str(i-8))
    #     plt.ylabel('Normalized Activation Strength for basis vector' + str(i-8) + ' of set 3')
    #     plt.xticks(rotation=45, ha='right')
    #     plt.xticks(ind, ('Bicep','Tricep lateral','Anterior deltoid','Medial deltoid','Posterior deltoid','Pectoralis major','Lower trapezius','Middle trapezius'))
    #     plt.legend(loc='best')
    # plt.title('Normalized Muscle synergy for set 3')

    # plt.savefig('/home/mushenghe/Desktop/final_project/muscle_synergy/src/image/Oct23/bootstrap_#2_set3.png')
    # plt.show()




    

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