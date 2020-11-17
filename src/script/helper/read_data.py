import numpy as np
import pandas as pd
import glob
from process_helper import *
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
    
    print('in read data')
    folders = glob.glob(
        path + '[c,s][0-9][0-9]/')  # returns all the folders in directory with format of 's%d' and 'c%d'
    print('28')
    print('folders',folders)
    if both_Arm:
        arms = ["Right", "Left"]
    else:
        arms = ["Right"]

    # initialize dataFrames in the returns
    col_name = ["ElTorque", "ShTorque",
                "Bicep", "Tricep", "AntDel", "MedDel",
                "PosDel", "Pec", "LowerTrap", "MidTrap",
                "Task", "TrialNr", 'Su', 'Arm']
    match_total = pd.DataFrame(columns=col_name)
    ref_total = pd.DataFrame(columns=col_name)
    demo_total = pd.DataFrame()

    for f in folders:  # based on the list of folders, step into each folder
        print('f',f)
        sname = f[-4:-1]
        for a in arms:
            print('a',a)
            # Obtain the baseline
            baseline_path = f + a + '/BaselineEMG/set01_trial01.txt'
            baseline_sit_path = f + a + '/BaselineEMG_sitting/set01_trial01.txt'
            baseline = compute_baseline_mean(baseline_path)
            print('baseline',baseline)
            baseline_sit = compute_baseline_mean(baseline_sit_path)  # baseline data for the matching trials
            baseline_maximum = baseline
            baseline_maximum[:1] = baseline_sit[:1]  # baseline data for maximum

            max_d = maximum_data(f + a, baseline_maximum)
            print(max_d)

            demo_total = demo_total.append(get_demo_info(f+a))

            ref_d, match_d = matching_trial_data(f + a, baseline_sit, max_d)
            print(match_d)
            print(ref_d)

            match_d['Su'], ref_d['Su'] = sname, sname
            match_d['Arm'], ref_d['Arm'] = a, a

        match_total = match_total.append(match_d)
        ref_total = ref_total.append(ref_d)


    print(ref_total)
    print(match_total)

    return ref_total, match_total, demo_total


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
        # add check for if file does not exist
        if len(max_trials) == 0:
            print("set ", i, " maximum value does not exist")
            continue
        for m in max_trials:
            data = np.loadtxt(m)
            trial_EMG = np.absolute(data[:, 10:] - baseline)
            trial_torque = data[:, 2:4]
            data_torEMG = np.concatenate((trial_torque, trial_EMG), axis=1)
            #  perform a moving average window on the target column within data_torEMG - non-weighted, window size 50

            after_maf = np.array([np.convolve(data_torEMG[:, col_idx], np.ones(250), 'valid') / 250
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
            ref_data, match_data: a 12-column dataframe
                    columns mapped as follows: 0-ElTorque, 1-ShTorque, 2-Bicep, 3-Tricep, 4-AntDel, 5-MedDel,
                                              6-PosDel, 7-Pec, 8-LowerTrap, 9-MidTrap, 10-Task, 11-TrialNr
        """
    # first, map the select_Trial code to task
    col_name = ["ElTorque", "ShTorque",
                "Bicep", "Tricep", "AntDel", "MedDel",
                "PosDel", "Pec", "LowerTrap", "MidTrap",
                "Task", "TrialNr"]
    task_dict = {1: '10%', 2: '30%', 3: '50%'}
    set_total_ref = np.zeros((30, 12))
    set_total_match = np.zeros((30, 12))
    for set_idx in range(3):
        stat_file = glob.glob(filepath + '/MatchingTask/*/trialDat_set0' + str(set_idx + 1) + '_trial0?.txt')[0]
        stat_data = np.loadtxt(stat_file)
        task_code = stat_data[9]

        # grab all the trials for this set
        trial_list = glob.glob(filepath + '/MatchingTask/*/set0' + str(set_idx + 1) + '_trial??.txt')
        for trial in trial_list:
            print("Reading file: ", trial)
            trial_nr = int(trial[-6:-4])
            trial_idx = (trial_nr - 1) + set_idx * 10
            trial_data = np.loadtxt(trial)
            # columns of trial data is as follows: 0-"simTime", 1-"State", 2-"Elbow_Torque", 3-"Shoulder_Torque",
            #                  4-"EFx", 5-"EFy", 6-"EFz", 7-"EMx", 8-"EMy", 9-"EMz",
            #                  10-"Bicep", 11-"Tricep", 12-"AntDel", 13-"MedDel",
            #                  14-"PosDel", 15-"Pec", 16-"LowerTrap", 17-"MidTrap"]

            # grab the start time of the reference phase:
            # make sure there is a reference phase
            if np.any(trial_data[:, 1] == 4):
                ref_idx = np.argwhere(trial_data[:, 1] == 4)
                # annotate the index
                ref_idx_s = ref_idx.max() - 999
                ref_idx_e = ref_idx.max() - 499
                ref_data = trial_data[ref_idx_s:ref_idx_e, :]
                ref_EMG = np.absolute(ref_data[:, 10:] - baseline).mean(axis=0)
                ref_EMG_norm = ref_EMG / max_data[2:]
                ref_torque = ref_data[:, 2:4].mean(axis=0)
                set_total_ref[trial_idx, :] = np.concatenate([ref_torque, ref_EMG_norm, [task_code, trial_nr]])
            else:
                continue
            if np.any(trial_data[:, 1] == 5) and (
                    np.argwhere(trial_data[:, 1] == 20).max() < np.argwhere(trial_data[:, 1] == 5).max()):
                match_idx = np.argwhere(trial_data[:, 1] == 5)
                # annote the index
                match_idx_s = match_idx.max() - 1249
                match_idx_e = match_idx.max() - 749
                match_data = trial_data[match_idx_s:match_idx_e, :]
                match_EMG = np.absolute(match_data[:, 10:] - baseline).mean(axis=0)
                match_EMG_norm = match_EMG / max_data[2:]
                match_torque = match_data[:, 2:4].mean(axis=0)
                set_total_match[trial_idx, :] = np.concatenate([match_torque, match_EMG_norm, [task_code, trial_nr]])
            else:
                continue

    ref_data_all = pd.DataFrame(set_total_ref, columns=col_name)
    match_data_all = pd.DataFrame(set_total_match, columns=col_name)
    ref_data = ref_data_all.replace(to_replace={"Task": task_dict})
    match_data = match_data_all.replace(to_replace={"Task": task_dict})
    print('finished')

    return ref_data.loc[~(ref_data == 0).all(axis=1)], match_data.loc[~(match_data == 0).all(axis=1)]


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
    ref_total, match_total, demo_total = read_data('/home/mushenghe/Desktop/final_project/data/', both_Arm=False)
    ref_total.to_csv('referenceData_111120.csv', index=False)
    match_total.to_csv('matchData_111120.csv', index=False)
    demo_total.to_csv('demoInfo_111120.csv', index=False)
