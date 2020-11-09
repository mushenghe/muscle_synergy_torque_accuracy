import numpy as np
import pandas as pd
import glob
from script.process_helper import *
import datetime
import matplotlib.pyplot as plt


def read_data(path, both_Arm=False):
    """

    Args:
        path: String - path to the directory that hosts the data
        both_Arm: Binary - Default to False meaning only right arm is tested and contains data
                            Set to True if both arms are tested and contain data

    Returns:
        max_EMGTorque: Data Frame - 8 columns of max muscle EMGs + 2 columns of max torques (EF & SABD)
        ref_EMGTorque: Data Frame - 8 columns of EMG and 2 columns of torque data during the reference phase
        match_EMGTorque: Data Frame - 8 columns of EMG and 2 columns of torque data during the matching phase
    """
    # When bootstrap methods are ready,
    # could return two more dataframe containing bootstrapped ref and match EMG torque data

    folders = glob.glob(
        path + '[c,s][0-9][0-9]/')  # returns all the folders in directory with format of 's%d' and 'c%d'
    if both_Arm:
        arms = ["Right", "Left"]
    else:
        arms = ["Right"]

    # initialize dataFrames in the returns
    df_data = {"Bicep": [], "Tricep": [], "AntDel": [], "MedDel": [],
               "PosDel": [], "Pec": [], "LowerTrap": [], "MidTrap": [],
               "ElTorque": [], "ShTorque": []}
    max_EMGTorque = pd.DataFrame(df_data)
    ref_EMGTorque = pd.DataFrame(df_data)
    match_EMGTorque = pd.DataFrame(df_data)

    # initialize column names for data
    stat_name = ["Year", "Month", "Day", "SuNr", "SuType", "TrialType", "Task", "SetCount", "TrialCount",
                 "SelectTrial", "TestArm", "Age", "Gender", "DomArm", "Handiness",
                 "Diabetes", "YearsSinceStroke", "MaxShoulderAb", "MaxExtension", "MaxFlexion", "Max4",
                 "ShoulderAbAngle", "ElbowFlexAngle", "Elbow_Humerus", "z-offset"]

    sens_name = ["simTime", "State", "Elbow_Torque", "Shoulder_Torque",
                 "EFx", "EFy", "EFz", "EMx", "EMy", "EMz",
                 "Bicep", "Tricep", "AntDel", "MedDel",
                 "PosDel", "Pec", "LowerTrap", "MidTrap"]

    col_name = ["ElTorque", "ShTorque",
                "Bicep", "Tricep", "AntDel", "MedDel",
                "PosDel", "Pec", "LowerTrap", "MidTrap"]
    for f in folders:  # based on the list of folders, step into each folder
        for a in arms:
            # Obtain the baseline
            baseline_path = f + a + '/BaselineEMG/set01_trial01.txt'
            baseline_sit_path = f + a + '/BaselineEMG_sitting/set01_trial01.txt'
            baseline = compute_baseline_mean(baseline_path)
            baseline_sit = compute_baseline_mean(baseline_sit_path)  # baseline data for the matching trials
            baseline_maximum = baseline
            baseline_maximum[:1] = baseline_sit[:1]  # baseline data for maximum

            print(maximum_data(f + a, baseline_maximum))


def maximum_data(filepath, baseline):
    """

    Args:
        filepath: str, path to the MaxMeasurements folder (directory + subject folder)
        baseline: 1d numpy array (size=8), baseline for maximum

    Returns:
        max_data: 1d numpy array (size=10), maximum torque + maximum EMG
                index mapping as follows: 0-ElTorque, 1-ShTorque, 2-Bicep, 3-Tricep, 4-AntDel, 5-MedDel,
                                          6-PosDel, 7-Pec, 8-LowerTrap, 9-MidTrap
    """
    max_dict = {6: [2, 0], 7: [3], 3: [4], 4: [5, 8],
                5: [6], 1: [7], 2: [9], 8: [1]}  # map set number to the column number of the corresponding target
    max_data = np.zeros(10)  # initiate the max_data array with a 1d numpy array of zeros
    for i in range(1, 9):  # i is the set count
        set_max_collector = []  # initiate set_max empty list to store trial_maxes within a set
        max_trials = glob.glob(filepath + '/MaxMeasurements/set0' + str(i) + '_trial0?.txt')
        for m in max_trials:
            data = np.loadtxt(m)
            trial_EMG = np.absolute(data[:, 10:] - baseline)
            trial_torque = data[:, 2:4]
            data_torEMG = np.concatenate((trial_torque, trial_EMG), axis=1)
            #  perform a moving average window on the target column within data_torEMG - non-weighted, window size 50

            after_maf = np.array([np.convolve(data_torEMG[:, col_idx], np.ones(50), 'valid') / 50
                                  for col_idx in max_dict[i]]).T
            trial_col_max = after_maf.max(axis=0).reshape(1, -1)
            set_max_collector.append(trial_col_max)
        set_max = np.array(set_max_collector).max(axis=0).flatten()
        for idx in range(len(max_dict[i])):
            print("max_dict[i][idx] is: ", max_dict[i][idx])
            print("set_max is: ", set_max)
            max_data[max_dict[i][idx]] = set_max[idx]
            print("set count is: ", i, "\nIdx is: ", idx)
    return max_data


def matching_trial_data(filepath, baseline, max_data):
    """

        Args:
            filepath: str, path to the MatchingTrial folder (directory + subject folder)
            baseline: 1d numpy array (size=8), sitting baseline
            max_data: 1d numpy array (size=10), maximum torque and EMG

        Returns:
            max_data: 1d numpy array (size=10), maximum torque + maximum EMG
                    index mapping as follows: 0-ElTorque, 1-ShTorque, 2-Bicep, 3-Tricep, 4-AntDel, 5-MedDel,
                                              6-PosDel, 7-Pec, 8-LowerTrap, 9-MidTrap
        """
    # first, map the select_Trial code to task
    task_dict = {1: '10%', 2: '30%', 3: '50%'}
    stat_file = glob.glob(filepath + '/MatchingTask/*/trialDat_set0?_trial0?.txt')[0]


    # return ref_data, match_data


def get_demo_info(filepath):
    """
    Load any one trialDat info in the MatchingTrial
    return dataframe: including experiment date, participant age, handedness,
                      dom/or non-paretic arm, arm_length, z-offset
    Args:
        filepath: str, path to the MatchingTrial folder

    Returns:
        demo_info: pandas dataframe

    """
    stat_name = ["Year", "Month", "Day", "SuNr", "SuType", "TrialType", "Task", "SetCount", "TrialCount",
                 "SelectTrial", "ArmTested", "SwitchSet", "Task", "Age", "Gender", "DomArm", "Handiness",
                 "Diabetes", "YearsSinceStroke", "MaxShoulderAb", "MaxExtension", "MaxFlexion", "Max4",
                 "ShoulderAbAngle", "ElbowFlexAngle", "Elbow_Humerus", "z-offset"]
    stat_file = glob.glob(filepath + '/MatchingTask/*/trialDat_set0?_trial0?.txt')[0]
    data = np.loadtxt(stat_file)  # 1d numpy array size 27
    stat_df = pd.DataFrame(data.reshape(1, -1), columns=stat_name)
    stat_df = stat_df.drop(["TrialType", "Task", "SetCount", "TrialCount",
                            "SelectTrial", "SwitchSet", "Task",
                            "Max4", "ShoulderAbAngle", "ElbowFlexAngle"], axis=1)
    demo_info = stat_df.replace(to_replace={'SuType': {1: 'c', 0: 's'}, 'ArmTested': {1: 'Right', 0: 'Left'},
                                            'Gender': {1: 'F', 0: 'M'}, 'DomArm': {1: 'Right', 0: 'Left'},
                                            'Handiness': {1: 'Right', 0: 'Left'}, 'Diabetes': {1: 'Y', 0: 'N'}})
    demo_info = demo_info.assign(Time=lambda x: pd.to_datetime(demo_info.iloc[:, 0:3])
                                 ).drop(["Year", "Month", "Day"], axis=1)
    return demo_info


if __name__ == "__main__":
    read_data('/Users/ncr5341/Downloads/', both_Arm=False)
