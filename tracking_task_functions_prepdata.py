import aopy
import numpy as np
import math as m
import collections
from datetime import datetime

from tracking_task_functions_bmi3d import *

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

def get_sequence_params(metadata):
    if 'sequence' not in metadata:
        raise ValueError("no sequence saved!")
    sequence = metadata['sequence']
    seq_name, seq_params = sequence.split('[')
    seq_dict = collections.defaultdict(list)
    for param in seq_params[:-1].split(', '):
        name, value = param.split('=')
        try:
            seq_dict[name] = int(value)
        except:
            if value == 'true':
                seq_dict[name] = True
            elif value == 'false':
                seq_dict[name] = False
            else:
                seq_dict[name] = np.fromstring(value, sep=' ')
                if len(seq_dict[name]) == 1:
                    seq_dict[name] = seq_dict[name][0]                    
    if 'ramp_down' not in seq_dict:
        seq_dict['ramp_down'] = 0
    return seq_name, seq_dict


def get_kinematic_segments_v4(datatype, trial_times, data, metadata):
    if datatype == 'cursor':
        data_cycles = data['cursor'][:,2] # 1d y (3d x,z,y)
    elif datatype == 'target':
        data_cycles = data['task']['current_target'][:,2] # 1d y
    elif datatype == 'disturbance':
        data_cycles = data['task']['current_disturbance'][:,2] # 1d y
    elif datatype == 'hand':
        data_cycles = data['task']['manual_input'] # 3d x,y,z
    elif datatype == 'input':
        # cursor position before disturbance added
        data_cycles = data['task']['manual_input'][:,1] + metadata['offset'][1] # 1d y
    elif datatype == 'generator':
        # generator index used to find ref_ind, dis_ind (freqs of each signal)
        data_cycles = data['task']['gen_idx']
        
    if np.isnan(data_cycles).all():
        print(datatype + ' is all nans')
        return np.empty((0,), dtype='object')
        
    clock = data['clock']['timestamp_sync']
    samplerate = metadata['fps']
    time = np.arange(int((clock[-1] + 10)*samplerate))/samplerate
    
    kinematics, _ = aopy.preproc.base.interp_timestamps2timeseries(clock, data_cycles, sampling_points=time, interp_kind='linear')
    trajectories = aopy.preproc.base.get_data_segments(kinematics, trial_times, samplerate)
    
    if len(trajectories) == 1:
        trajectories = np.array(trajectories)
        corr_trajectories = np.empty((1,), dtype='object')
        corr_trajectories[0] = np.squeeze(trajectories)    
    else:
        corr_trajectories = np.empty(len(trajectories), dtype='object')
        corr_trajectories[:] = trajectories

    return corr_trajectories


def get_kinematic_segments_v5(datatype, trial_times, data, metadata, eye_data, eye_metadata): 
    if datatype == 'cursor':
        data_cycles = data['task']['cursor'][:,2] # 1d y (3d x,z,y)
    elif datatype == 'target':
        data_cycles = data['task']['current_target'][:,2] # 1d y
    elif datatype == 'disturbance':
        data_cycles = data['task']['current_disturbance'][:,2] # 1d y
    elif datatype == 'hand':
        data_cycles = data['task']['manual_input'] # 3d x,y,z
    elif datatype == 'input':
        # cursor position before disturbance added
        data_cycles = data['task']['manual_input'][:,1] + metadata['offset'][1] # 1d y
    elif datatype == 'generator':
        # generator index used to find ref_ind, dis_ind (freqs of each signal)
        data_cycles = data['task']['gen_idx']
    elif datatype == 'eye': # 4d Lx, Ly, Rx, Ry
        if 'calibrated_data' in eye_data:
            data_cycles = eye_data['calibrated_data']
        else:
            print(datatype + 'is not calibrated')
            return np.empty((0,), dtype='object')

    if np.isnan(data_cycles).all():
        print(datatype + ' is all nans')
        return np.empty((0,), dtype='object')
    
    clock = data['clock']['timestamp_sync']
    samplerate = metadata['fps']
    time = np.arange(int((clock[-1] + 10)*samplerate))/samplerate
    if datatype == 'eye':
        time = np.arange(len(data_cycles))/eye_metadata['samplerate']
        kinematics, _ = aopy.preproc.base.interp_timestamps2timeseries(time, data_cycles, samplerate)    
    else:
        kinematics, _ = aopy.preproc.base.interp_timestamps2timeseries(clock, data_cycles, sampling_points=time)
        
    trajectories = aopy.preproc.base.get_data_segments(kinematics, trial_times, samplerate)
    
    if len(trajectories) == 1:
        trajectories = np.array(trajectories)
        corr_trajectories = np.empty((1,), dtype='object')
        corr_trajectories[0] = np.squeeze(trajectories)    
    else:
        corr_trajectories = np.empty(len(trajectories), dtype='object')
        corr_trajectories[:] = trajectories

    return corr_trajectories


def get_reward_trials_and_pairs(exp_data, exp_metadata, primes, eye_data=None, eye_metadata=None):
    event_codes = exp_data['events']['code']
    event_times = exp_data['events']['timestamp'] 
    
    # GET FREQUENCY INDICES FOR ALL TRIALS
    # generate reference and disturbance trajectories
    seq_name, seq_dict = get_sequence_params(exp_metadata)
    num_trials = seq_dict['ntrials']
    time_length = seq_dict['time_length']
    ramp = seq_dict['ramp']
    num_primes = seq_dict['num_primes']
    seed = seq_dict['seed']
    sample_rate = seq_dict['sample_rate']
    ramp_down = seq_dict['ramp_down']

    gen_trials, gen_trial_order = generate_trajectories(num_trials=num_trials, time_length=time_length, seed=seed, \
                                                sample_rate=sample_rate, ramp=ramp, ramp_down=ramp_down, num_primes=num_primes)

    # create list of which frequency indices belong to reference and disturbance
    sines_r = []
    sines_d = []

    for trial_id, (num_reps,ref_ind,dis_ind) in enumerate(gen_trial_order*int(num_trials/2)):      
        if ref_ind == 'E':
            sines_r.append(np.arange(len(primes))[0::2]) # use even indices
        elif ref_ind == 'O': 
            sines_r.append(np.arange(len(primes))[1::2]) # use odd indices
        else:
            sines_r.append(np.arange(len(primes)))
        if dis_ind == 'E':
            sines_d.append(np.arange(len(primes))[0::2])
        elif dis_ind == 'O':
            sines_d.append(np.arange(len(primes))[1::2])
        else:
            sines_d.append(np.arange(len(primes)))

    # PREPARE DATA DIFFERENTLY DEPENDING ON EXPERIMENT DATE
    date = datetime.strptime(exp_metadata['date'].split(' ')[0], '%Y-%m-%d')
    
    if date < date.fromisoformat('2023-02-23'): # BEFORE generator index was saved in 'task' on every cycle
        print('...using segments and gen inds from all trials to get those for reward trials...')
        
        # GET ALL TRIAL SEGMENTS 
        trial_start_codes = [TARGET_ON]
        trial_end_codes = [REWARD, TIMEOUT_PENALTY, HOLD_PENALTY, OTHER_PENALTY] # TRIAL_END alone may not get every trial 

        trial_segments, trial_times = aopy.preproc.base.get_trial_segments(event_codes, event_times, trial_start_codes, trial_end_codes)
        _, trial_times_all = aopy.preproc.base.get_trial_segments_and_times(event_codes, event_times, trial_start_codes, trial_end_codes)

        trial_segments = np.array(trial_segments,dtype=np.ndarray)
        trial_times_all = np.array(trial_times_all,dtype=np.ndarray)

#         assert len(trial_segments) == num_total
        print(len(trial_segments), 'total trials')
        
        # GET ALL TRIAL GENERATOR INDICES
        max_holds = exp_metadata['max_hold_attempts']
        trial_gen_indices = [] # used to index into any generator-related matrix

        num_holds = 0
        gen_idx = 0

        for trial_num in range(len(trial_segments)):
            trial_gen_indices.append(gen_idx)

            if CURSOR_ENTER_TARGET in trial_segments[trial_num]:
                num_holds = 0
            if HOLD_PENALTY in trial_segments[trial_num]:
                num_holds +=1

            if num_holds==0 or num_holds>=max_holds:
                gen_idx +=1

        assert len(trial_gen_indices) == len(trial_segments)
        
        # GET REWARD TRIAL SEGMENTS
        reward_ids = []
        for trial_id,segment in enumerate(trial_segments):
            if REWARD in segment:
                reward_ids.append(trial_id)
        print(len(reward_ids), 'reward trials')
        
        reward_segments = trial_segments[reward_ids]
        reward_times = trial_times[reward_ids]
        reward_times_all = trial_times_all[reward_ids]
        
        # get reward times specifically around tracking (from first CURSOR_ENTER_TARGET to REWARD)
        reward_times_tracking = []
        for trial_id,segment in enumerate(reward_segments):
            start_ind = np.where(segment==CURSOR_ENTER_TARGET)[0][0]
            end_ind = -1
            reward_times_tracking.append([ reward_times_all[trial_id][start_ind]+ramp, 
                                           reward_times_all[trial_id][end_ind]-ramp_down ])
        reward_times_tracking = np.array(reward_times_tracking)
        
        # GET REWARD TRIAL GENERATOR INDICES - used to index into any generator-related matrix
        trial_gen_indices = np.array(trial_gen_indices)
        reward_gen_indices = trial_gen_indices[reward_ids]
        
    else: # AFTER generator index was saved in 'task' on every cycle
        print('...getting segments and gen inds directly for reward trials...')
        
        # GET REWARD TRIAL SEGMENTS
        trial_start_codes = [TRIAL_START]
        trial_end_codes = [REWARD] 

        reward_segments, reward_times = aopy.preproc.base.get_trial_segments(event_codes, event_times, trial_start_codes, trial_end_codes)
        _, reward_times_all = aopy.preproc.base.get_trial_segments_and_times(event_codes, event_times, trial_start_codes, trial_end_codes)

        reward_segments = np.array(reward_segments,dtype=np.ndarray)
        reward_times_all = np.array(reward_times_all,dtype=np.ndarray)

#         assert len(reward_segments) == num_reward
        print(len(reward_segments), 'reward trials')
        
        # get reward times specifically around tracking (from first CURSOR_ENTER_TARGET to REWARD)
        reward_times_tracking = []
        for trial_id,segment in enumerate(reward_segments):
            start_ind = np.where(segment==CURSOR_ENTER_TARGET)[0][0]
            end_ind = -1
            reward_times_tracking.append([ reward_times_all[trial_id][start_ind]+ramp, 
                                           reward_times_all[trial_id][end_ind]-ramp_down ])
        reward_times_tracking = np.array(reward_times_tracking)
    
        # GET REWARD TRIAL GENERATOR INDICES - used to index into any generator-related matrix
        gen_indices = get_kinematic_segments_v4('generator', reward_times_tracking, exp_data, exp_metadata)
        reward_gen_indices = [int(gen[0]) for gen in gen_indices]
 
    
    # GET ALL RELEVANT TRAJECTORIES FOR REWARD TRIALS
    # target trajectories (r)
    ref = get_kinematic_segments_v5('target', reward_times_tracking, exp_data, exp_metadata, eye_data, eye_metadata)

    # disturbance trajectories (d)
    dis = get_kinematic_segments_v5('disturbance', reward_times_tracking, exp_data, exp_metadata, eye_data, eye_metadata)

    # manual input trajectories (u) 
    user = get_kinematic_segments_v5('input', reward_times_tracking, exp_data, exp_metadata, eye_data, eye_metadata)

    # cursor trajectories (y)
    cursor = get_kinematic_segments_v5('cursor', reward_times_tracking, exp_data, exp_metadata, eye_data, eye_metadata)
    
    # 3d hand trajectories - ADDED 3/20/23
    hand = get_kinematic_segments_v5('hand', reward_times_tracking, exp_data, exp_metadata, eye_data, eye_metadata)
    
    if eye_data is not None:
        # 2d eye trajectories - ADDED 4/9/23
        eye = get_kinematic_segments_v5('eye', reward_times_tracking, exp_data, exp_metadata, eye_data, eye_metadata)
    
    # check for any nans in the manual input (intended cursor position)
    for i in range(len(user)):
        if np.isnan(user[i]).any():
            print('Nans in manual input, trial', i)

    # set disturbance to 0 if disturbance was turned off
    if not seq_dict['disturbance']:
        dis = 0*dis


    # GET FREQUENCY INDICES FOR REWARD TRIALS
    # get frequency indices for reward trials - used to index into primes (i.e. frequencies)
    sines_r = np.array(sines_r)
    sines_d = np.array(sines_d)
    reward_sines_r = sines_r[reward_gen_indices]
    reward_sines_d = sines_d[reward_gen_indices]
    
    
    # PAIR UP EVEN/ODD REWARD TRIALS FOR CONTROLLER ANALYSIS - used to index into reward trials
    reward_sines_pairs = []
    counts = np.ones(len(reward_sines_r),)*2
    for i in range(len(reward_sines_r)):
        curr_trial = reward_sines_r[i]

        for j in range(i+1,len(reward_sines_r)):
            next_trial = reward_sines_r[j]
            if next_trial[0] != curr_trial[0] and j-i<3: # this is arbitrary - is it necessary?
                reward_sines_pairs.append([i,j])
                counts[i] -= 1
                counts[j] -= 1
                break

    leftover_trials = np.where(counts>0)[0]
    if len(leftover_trials)>0:
        for idx,i in enumerate(leftover_trials):
            curr_trial = reward_sines_r[i]

            for next_idx,j in enumerate(leftover_trials[idx+1:]):
                next_trial = reward_sines_r[j]
                if next_trial[0] != curr_trial[0] and j-i<3: # this is arbitrary - is it necessary?
                    reward_sines_pairs.append([i,j])

    reward_sines_pairs = sorted(reward_sines_pairs, key=lambda pair: pair[1]) # sort the pairs so trials retain order
    print(len(reward_sines_pairs), 'controllers')
        
        
    # SAVE TRIALS DICTIONARY
    reward_trials = []
    for i in range(len(user)):
        trial = dict(ref=ref[i], dis=dis[i], input=user[i], state=cursor[i], hand=hand[i],
                     ref_ind=reward_sines_r[i], dis_ind=reward_sines_d[i])
        if eye_data is not None:
            trial['eye']=eye[i]
        reward_trials.append(trial)
    
    return seq_dict, reward_trials, reward_sines_pairs, reward_sines_r, reward_sines_d