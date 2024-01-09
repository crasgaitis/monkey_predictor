import numpy as np
import aopy
import pandas as pd
import os
from matplotlib import animation
import datetime
from itertools import chain
import torch
import tensorflow as tf
from torch.utils.data import TensorDataset, DataLoader, random_split
from keras.layers import SimpleRNN
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# from db import dbfunctions as db
import re
import ast

from tracking_task_functions_bmi3d import *
from tracking_task_functions_analysis import *
from tracking_task_functions_prepdata import *

# set up constants
subject = 'churro'
data_dir = '/data/raw'
preproc_dir = '/data/preprocessed/'
headstage_voltsperbit = 1.907348633e-7

TARGET_ON = 16
TRIAL_START = 2 # start hold
CURSOR_ENTER_TARGET = 80 # 1st time/trial ONLY: start tracking
CURSOR_LEAVE_TARGET = 96
REWARD = 48
TIMEOUT_PENALTY = 65 # failure to opt in
HOLD_PENALTY = 64 # failure to complete hold
OTHER_PENALTY = 79 # tracking out penalty
PAUSE = 254
TIME_ZERO = 238
TRIAL_END = 239 # number of occurrences should match TARGET_ON
# TRIAL_END event is omitted when PAUSE occurs after OTHER_PENALTY

primes = np.asarray([2, 3, 5, 7, 11, 13, 17, 19])

penalty_map = {
    65: "Timeout Penalty (failure to opt in)",
    64: "Hold Penalty (failure to complete hold)",
    79: "Other Penalty (tracking out penalty)"
}

def segment(start_events, end_events, data):
    # TAKEN FROM get_trial_segments_and_times
    # Find the indices in events that correspond to start events

    bmi3d_events = data['bmi3d_events']
    # sync_events = data['sync_events'] - does not exist in raw data

    events = bmi3d_events
    event_code = events['code']
    event_inds = events['time']
    
    evt_start_idx = np.where(np.in1d(event_code, start_events))[0]

    # Extract segments for each start event
    segments = []
    segment_times = []
    for idx_evt in range(len(evt_start_idx)):
        idx_start = evt_start_idx[idx_evt]
        idx_end = evt_start_idx[idx_evt] + 1

        # Look forward for a matching end event
        while idx_end < len(event_code):
            if np.in1d(event_code[idx_end], start_events):
                break # start event must be followed by end event otherwise not valid
            if np.in1d(event_code[idx_end], end_events):
                segments.append(event_code[idx_start:idx_end+1])
                segment_times.append(event_inds[idx_start:idx_end+1])
                break
            idx_end += 1
            
    return segments, segment_times

def sort_by_te_number(file_names_filtered):
    return int(file_names_filtered.split('_te')[1].split('.')[0])

# get files after 1/20/23
def grab_files(data_dir = '/media/moor-data/tablet/hdf', prefix = 'chur', start_date_str = '20230121'):

    file_names = [f for f in os.listdir(data_dir) if f.startswith(prefix+'20')]

    file_names_filtered = []
    for name in file_names:
        date_str = name.split('_')[0].replace(prefix, '')
        if date_str >= start_date_str:
            file_names_filtered.append(name)

    print(file_names_filtered)
    file_names_sorted = sorted(file_names_filtered, key=sort_by_te_number)
    
    print(f'{(len(file_names_sorted))} files parsed.')
    
    print(file_names_sorted[0])
    print(file_names_sorted[-1])

    # get dates since prev
    dates = [name.split('_')[0].replace(prefix, '') for name in file_names_sorted]
    dates = [datetime.strptime(d, '%Y%m%d').date() for d in dates]
    prev_date = None
    deltas = []

    for date in dates:
        if prev_date:
            delta = (date - prev_date).days
        else:
            delta = 0
        deltas.append(delta)
        prev_date = date

    df = pd.DataFrame({'File Name': file_names_sorted, 'Days Since Prev': deltas})
 
    return df, file_names_sorted


# set up csv for hdf files
def build_file_df(df, file_names_sorted, data_dir = '/media/moor-data/tablet/hdf', traj = True):

    time2=5
    print('Building dataset... \n Problematic files:')
    
    for i, f in enumerate(file_names_sorted):
        files = dict(hdf=f)
        data, metadata = aopy.preproc.bmi3d._parse_bmi3d_v1(data_dir, files)
        bmi3d_task = data['bmi3d_task']

        # get metadata (built in)
        pattern = r'"runtime":\s*([0-9.]+)'

        try:
            match = re.search(pattern, metadata['report'])
            runtime = round(float(match.group(1)))

        except:
            print(f)
            continue
        
        df.loc[i, 'Runtime'] = runtime
        
        # df.loc[i, 'Target radius'] = metadata['target_radius']        
        # df.loc[i, 'Cursor radius'] = metadata['cursor_radius']
        
        if traj:
            try:
                df.loc[i, 'Trajectory Amplitude'] = np.ceil(max(abs(min(bmi3d_task['current_target'][:, 2])), abs(max(bmi3d_task['current_target'][:, 2]))))
            except:
                print(f)
                continue
            
        # trial length
        rewarded = segment([TRIAL_START], [REWARD], data)[1]
        time = []
        for j in (rewarded):
            time.append(j[-1] - j[0])
            time2 = round((np.median(time))/metadata['fps'])
    
        df.loc[i, 'Reward trial length'] = time2
    return df
    
    
# consolidate filenames on same day
def same_day_fix(df, data_dir = '/media/moor-data/tablet/hdf'):
    
    check_file_name2 = 'na'
    runtime = 0
    runtimes = []
    ntrial = 0
    ntrials = []
    reward_len = 0
    reward_lens = []
    f_s = []
    f_actual = []
    reward_time_check = []

    for i, f in enumerate(df['File Name']):
        files = dict(hdf=f)
        data, metadata = aopy.preproc.bmi3d._parse_bmi3d_v1(data_dir, files)
        
        match = re.search(r'(\d{4})(\d{2})(\d{2})_(\d{2})_te\d+\.hdf', f)
        check_file_name = '{}_{}'.format(match.group(2), match.group(3))
        
        row = df.loc[df['File Name'] == f]
        match2 = re.search(r'"n_trials"\s*:\s*(\d+)', metadata['report'])
        
        if check_file_name == check_file_name2:
            runtimes.pop()
            runtime += float(row['Runtime'].values[0])  
            
            ntrials.pop()
            ntrial += float(match2.group(1))
            
            reward_lens.pop()
            reward_len = float(row['Reward trial length'].values[0]) 
            
            f_s.pop() 
            
            f_actual.pop()     

        else:
            runtime = float(row['Runtime'].values[0])
            ntrial = float(match2.group(1))
            reward_len = float(row['Reward trial length'].values[0])  

        reward_time_check.append(metadata['reward_time'])
        runtimes.append(runtime)
        ntrials.append(ntrial)
        reward_lens.append(reward_len)
        f_s.append(check_file_name)
        f_actual.append(f)
        
        check_file_name2 = check_file_name
                
        # print(f'{sum(runtimes)/60} hours of training total')
        # print(f'{sum(ntrials)} trials total')
        
        return df
        
        
def remove_weird_days(df):
    df.drop(df[df['Runtime'] <= 10].index, inplace = True)
    df.reset_index(drop=True, inplace=True)
    return df

def build_hdf_df(df, file_names_sorted, data_dir='/media/moor-data/tablet/hdf', traj=True):
    df = build_file_df(df, file_names_sorted, data_dir, traj)
    df = same_day_fix(df, data_dir)
    df = remove_weird_days(df)
    return df

def get_target_cursor_info(segment_times, metadata, bmi3d_task, dir = "Y"):
    
    val = 0 if dir == "X" else 2
    
    hold = int(metadata['hold_time']*metadata['fps']) # number of bmi3d cycles
    target = bmi3d_task['current_target'][:,val]
    cursor = bmi3d_task['cursor'][:,val]
    target_segments = []
    cursor_segments = []

    for segment_time in segment_times:
        target_segments.append(target[int(segment_time[0])+hold:int(segment_time[-1])])
        cursor_segments.append(cursor[int(segment_time[0]+hold):segment_time[-1]])
        
    
    return target_segments, cursor_segments


# make csv for trials
def build_trial_df_from_hdf_df(df, data_dir = '/media/moor-data/tablet/hdf'):

    df2 = pd.DataFrame()

    for i, f in enumerate(df['File Name']):
        files = dict(hdf=f)
        data, metadata = aopy.preproc.bmi3d._parse_bmi3d_v1(data_dir, files)
        bmi3d_task = data['bmi3d_task']
        
        events, times = segment([TRIAL_START], [REWARD, OTHER_PENALTY], data)

        set = df2.shape[0]
        
        for j in range(len(events)):
                
            # trial naming
            df2.loc[j+set, 'Trial'] = (f.split('_te')[1].split('.')[0]) + "_" + str(j)
            
            # get df2 col relevant metadata
            df2.loc[j+set, 'Days Since Prev'] = df.loc[df['File Name'] == f, "Days Since Prev"].iloc[0]

            # end states
            
            e = events[j] #bmi3d_events
            val = 1 if e[-1] == REWARD else 0
            df2.loc[j+set, 'End state'] = val
            
            # trial length (total)
            t = times[j]
            t2 = (t[-1] - t[0]) / metadata['fps']
            df2.loc[j+set, 'Time'] = round(t2, 2)
            
            # get tracking time
            flag = 0
            enter_times = [] 
            for l, k in enumerate(e):
                if k in [CURSOR_ENTER_TARGET, TARGET_ON]:
                    enter_time = t[l]
                    flag = 1
                
                if k in [CURSOR_LEAVE_TARGET, REWARD, OTHER_PENALTY] and flag == 1:
                    leave_time = t[l]
                    flag = 0
                    
                    enter_times.append((leave_time - enter_time) / metadata['fps'])

            target_segments, cursor_segments = get_target_cursor_info(np.array([t]), metadata, bmi3d_task)
                
            # get error y dir
            error = [
                [c - t for c, t in zip(cursor_segment, target_segment)]
                for cursor_segment, target_segment in zip(cursor_segments, target_segments)
            ]

            # error = [np.round(value, 2) for value in error] # round

            df2.loc[j+set, 'Tracking times'] = sum(enter_times)  
            
            # get error lists
            df2.loc[j+set, 'Error list'] = str(error)
        
    # df2 = df2.drop(columns=["Target radius", "Cursor radius"])            
    return df2

# add prefix and suffix
def add_prefix_suffix(df2):
    df2['Prefix'] = df2['Trial'].apply(lambda x: x.split('_')[0])
    df2['Suffix'] = df2['Trial'].apply(lambda x: x.split('_')[1])
    return df2

# build dataframe to represent each hdf file
def build_df_for_all_hdfs(df, file_names_sorted):
    df = build_file_df(df, file_names_sorted)
    df = same_day_fix(df)
    df = remove_weird_days(df)
    return df

# add summary stats using hdf/session-level and trial-level data
def add_hdf_summary_stats(df1, df2, traj=True):
    # ID extracted from "File Name"
    df1['ID'] = df1['File Name'].apply(lambda x: re.search(r'_te(\d+).hdf', x).group(1) if re.search(r'_te(\d+).hdf', x) else None)

    # initialize new columns for df1
    df1['Tracking time %'] = 0.0
    df1['Reward %'] = 0.0

    # iterate through rows in df1
    for index, row in df1.iterrows():
        # extract ID from the current row
        current_id = row['ID']

        # select rows from df2 based on the extracted ID
        selected_rows_df2 = df2[df2['Trial'].str.split('_').str[0] == current_id]

        # calculate "Tracking time %"
        tracking_time_sum = selected_rows_df2['Tracking times'].sum()
        time_sum = selected_rows_df2['Time'].sum()
        tracking_time_percent = (tracking_time_sum / time_sum) * 100

        # calculate "Reward %"
        reward_percent = (selected_rows_df2['End state'].sum() / len(selected_rows_df2)) * 100

        # update df1 with calculated values
        df1.at[index, 'Tracking time %'] = tracking_time_percent
        df1.at[index, 'Reward %'] = reward_percent
        
        # add a new column "Prev_Same_Sessions"
        if traj:
            prev_same_sessions_count = len(df1[(df1['Trajectory Amplitude'] == row['Trajectory Amplitude']) & (df1['Reward trial length'] == row['Reward trial length']) & (df1.index < index)])
        else:
            prev_same_sessions_count = len(df1[(df1['Reward trial length'] == row['Reward trial length']) & (df1.index < index)])

        df1.at[index, 'Prev_Same_Sessions'] = prev_same_sessions_count

        # extract error lists from selected rows in df2
        error_list_column = selected_rows_df2['Error list']
        error_lists = error_list_column.apply(ast.literal_eval)

        # flatten the lists and take the absolute values
        flattened_array = np.concatenate(error_lists.tolist(), axis=1)
        flattened_array = np.abs(flattened_array)[0]

        # define bins
        bins = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, np.inf]
        bin_labels = ["0_to_0.5", "0.5_to_1", "1_to_1.5", "1.5_to_2", "2_to_3", "3_to_4", "4_to_5", "5_to_7", "7_to_10", "over_10"]

        # count occurrences in each bin
        bin_counts = np.histogram(flattened_array, bins=bins)[0]

        # add new columns to df1 based on bins
        for label, count in zip(bin_labels, bin_counts):
            percentage = (count / len(flattened_array)) * 100
            df1.at[index, label] = percentage

    df1.drop('ID', axis=1, inplace=True)

    return df1

# make a prediction w/ model
def make_pred(prefix, model, difficulty=None):
    
    subj, subj_file_names_sorted = grab_files(show=False, prefix = prefix)
    subj2 = build_hdf_df(subj, subj_file_names_sorted)
    subj3 = build_trial_df_from_hdf_df(subj2)        

    X = add_hdf_summary_stats(subj2, subj3)

    X_row = X.iloc[[-1]]    
    X_row = X_row.drop(columns=['File Name', 'Reward trial length', 'Trajectory Amplitude'])
    
    if difficulty != None:
        X_row['t_update'] = difficulty
    
    pred = model.predict(X_row)
    
    return pred

# add labels to dataset
def add_labels(df, traj=True):
    # initialize new columns
    df['t_reward'] = np.nan
    df['t_track'] = np.nan
    df['t_update'] = np.nan

    for index in range(len(df) - 1):
        # get values for the next row
        next_row = df.iloc[index + 1]

        # set t_reward and t_track for all rows except the last one
        df.at[index, 't_reward'] = next_row['Reward %']
        df.at[index, 't_track'] = next_row['Tracking time %']

        # set t_update based on conditions
        if traj:
            if (df.at[index, 'Trajectory Amplitude'] == next_row['Trajectory Amplitude']) and \
                    (df.at[index, 'Reward trial length'] == next_row['Reward trial length']):
                df.at[index, 't_update'] = 0
                
            elif (next_row['Trajectory Amplitude'] < df.at[index, 'Trajectory Amplitude']) and \
                    (next_row['Reward trial length'] < df.at[index, 'Reward trial length']):
                df.at[index, 't_update'] = -1
                
            else:
                df.at[index, 't_update'] = 1
                
        else:
            if (df.at[index, 'Reward trial length'] == next_row['Reward trial length']):
                df.at[index, 't_update'] = 0
            
            elif (next_row['Reward trial length'] < df.at[index, 'Reward trial length']):
                df.at[index, 't_update'] = -1
                
            else:
                df.at[index, 't_update'] = 1              

    return df