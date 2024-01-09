import numpy as np

def calc_sum_of_sines(times, frequencies, amplitudes, phase_shifts):
    '''
    Generates the trajectories for the experiment
    '''
    t = times
    t = np.asarray(t).copy(); t.shape = (t.size,1)

    f = frequencies
    f = f.copy()
    f = np.reshape(f, (1,f.size))

    a = amplitudes
    a = a.copy()
    a = np.reshape(a, (1,a.size))

    p = phase_shifts        
    p = p.copy()
    p = np.reshape(p, (1,p.size))

    assert f.shape == a.shape == p.shape,"Shape of frequencies, amplitudes, and phase shifts must match"

    o = np.ones(t.shape)
    trajectory = np.sum(np.dot(o,a) * np.sin(2*np.pi*(np.dot(t,f) + np.dot(o,p))),axis=1)
    A = np.sum(np.dot(o,a)[0])

    return trajectory, A

def calc_sum_of_sines_ramp(times, ramp, ramp_down, frequencies, amplitudes, phase_shifts):
        '''
        Adds a 1/t ramp up and ramp down at the start and end so the trajectories start and end at zero.
        '''
        t = times
        t = np.asarray(t).copy(); t.shape = (t.size,1)

        r = ramp
        rd = ramp_down

        trajectory, A = calc_sum_of_sines(t, frequencies, amplitudes, phase_shifts)

        if r > 0:
            trajectory *= (( t*(t <= r)/r + (t > r) ).flatten())**2
            # trajectory *= (((t*(t <= r)/r) + ((t > r) & (t < (t[-1]-r))) + ((t[-1]-t)*(t >= (t[-1]-r))/r)).flatten())**2

        if rd > 0:
            trajectory *= (( (t < (t[-1]-rd)) + ((t[-1]-t)*(t >= (t[-1]-rd))/rd) ).flatten())**2

        return trajectory, A

def generate_trajectories(num_trials=2, time_length=20, seed=40, sample_rate=120, base_period=20, ramp=0, ramp_down=0, num_primes=8):
    '''
    Sets up variables and uses prime numbers to call the above functions and generate then trajectories
    ramp is time length for preparatory lines
    '''
    np.random.seed(seed)
    hz = sample_rate # Hz -- sampling rate
    dt = 1/hz # sec -- sampling period

    T0 = base_period # sec -- base period
    w0 = 1./T0 # Hz -- base frequency

    r = ramp # "ramp up" duration (see sum_of_sines_ramp)
    rd = ramp_down # "ramp down" duration (see sum_of_sines_ramp)
    P = time_length/T0 # number of periods in signal
    T = P*T0+r+rd # sec -- signal duration
    dw = 1./T # Hz -- frequency resolution
    W = 1./dt/2 # Hz -- signal bandwidth

    full_primes = np.asarray([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 
        101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199])
    primes = full_primes[:num_primes]
    # primes_ind = np.where(full_primes <= T0)
    # primes = full_primes[primes_ind]

    f = primes*w0 # stimulated frequencies
    f_ref = f.copy()
    f_dis = f.copy()

    a = 1/(1+np.arange(f.size)) # amplitude
    a_ref = a.copy()
    a_dis = a.copy()

    o = np.random.rand(num_trials,primes.size) # phase offset
    o_ref = o.copy()
    o_dis = o.copy()*0.8

    t = np.arange(0,T,dt) # time samplesseed
    w = np.arange(0,W,dw) # frequency samples

    N = t.size # = T/dt -- number of samples

    # create trials dictionary
    trials = dict(
        id=np.arange(num_trials), times=np.tile(t,(num_trials,1)), ref=np.zeros((num_trials,N)), dis=np.zeros((num_trials,N))
        )

    # randomize order of first two trials to provide random starting point
    order = np.random.choice([0,1])
    if order == 0:
        trial_order = [(1,'E','O'),(1,'O','E')]
    elif order == 1:
        trial_order = [(1,'O','E'),(1,'E','O')]

    # generate reference and disturbance trajectories for all trials
    for trial_id, (num_reps,ref_ind,dis_ind) in enumerate(trial_order*int(num_trials/2)):   
        if ref_ind == 'E': 
            sines_r = np.arange(len(primes))[0::2] # use even indices
        elif ref_ind == 'O': 
            sines_r = np.arange(len(primes))[1::2] # use odd indices
        else:
            sines_r = np.arange(len(primes))
        if dis_ind == 'E':
            sines_d = np.arange(len(primes))[0::2]
        elif dis_ind == 'O':
            sines_d = np.arange(len(primes))[1::2]
        else:
            sines_d = np.arange(len(primes))

        ref_trajectory, ref_A = calc_sum_of_sines_ramp(t, r, rd, f_ref[sines_r], a_ref[sines_r], o_ref[trial_id][sines_r])
        dis_trajectory, dis_A = calc_sum_of_sines_ramp(t, r, rd, f_dis[sines_d], a_dis[sines_d], o_dis[trial_id][sines_d])

        # normalized trajectories
        trials['ref'][trial_id] = ref_trajectory/ref_A   # previously, denominator was np.sum(a_ref)
        trials['dis'][trial_id] = dis_trajectory/dis_A   # previously, denominator was np.sum(a_dis)

    return trials, trial_order


# two ways to normalize the trajectories' amplitudes
# len(exp_data['bmi3d_task']['trial'])
# a = 1/(1+np.arange(8))
# o_a = a[1::2]
# e_a = a[::2]
# e_a/np.sum(e_a)*2.5, o_a/np.sum(o_a)*2.5
# print(np.sum(e_a/np.sum(e_a)), np.sum(o_a/np.sum(o_a)))

# np.sum(a)
# e_a/np.sum(a)*2.5, o_a/np.sum(a)*2.5
# print(np.sum(e_a/np.sum(a)), np.sum(o_a/np.sum(a)))