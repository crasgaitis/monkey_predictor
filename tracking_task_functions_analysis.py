import numpy as np
import math as m

def findFFT(primes, fs, base_period, n_base_cycles, trials):
    '''
    returns time and frequency domain values for each trial in trials
    '''
    n = int(base_period*n_base_cycles*fs) # number of samples (total trial time * sample rate)
    # actual number of samples may differ from n due to imperfect bmi3d framerate

    xf_all = np.fft.fftfreq(n, 1./fs)       # freq (x-axis) both + and - terms
    xf = np.fft.fftfreq(n, 1./fs)[:n//2]    # freq (x-axis) positive-frequency terms

    IX = primes*n_base_cycles

    tempfreq = np.ones((n,))
    TEMPFREQ = np.fft.fft(tempfreq)[0]

    timedomainvalues = {}
    for id, trial in enumerate(trials):
        timedomainvalues[id] = {}
        timedomainvalues[id]['ref'] = trial['ref'][:n]
        timedomainvalues[id]['dis'] = trial['dis'][:n]
        timedomainvalues[id]['input'] = trial['input'][:n] # cursor position before disturbance added
        timedomainvalues[id]['state'] = trial['state'][:n] # cursor position

    FREQDOMAINVALUES = {}
    for id, trial in enumerate(trials):
        # perform FFT
        FREQDOMAINVALUES[id] = {}
        FREQDOMAINVALUES[id]['REF'] = np.fft.fft(timedomainvalues[id]['ref'])[:int(n/2)]/TEMPFREQ # positive-frequency terms
        FREQDOMAINVALUES[id]['DIS'] = np.fft.fft(timedomainvalues[id]['dis'])[:int(n/2)]/TEMPFREQ
        FREQDOMAINVALUES[id]['INPUT'] = np.fft.fft(timedomainvalues[id]['input'])[:int(n/2)]/TEMPFREQ
        FREQDOMAINVALUES[id]['STATE'] = np.fft.fft(timedomainvalues[id]['state'])[:int(n/2)]/TEMPFREQ

        # calculate freq domain transfer functions --> eq. 5 in Yamagami 2021 paper       
        FREQDOMAINVALUES[id]['Tur'] = np.divide(np.squeeze(FREQDOMAINVALUES[id]['INPUT'][IX]),FREQDOMAINVALUES[id]['REF'][IX])
        FREQDOMAINVALUES[id]['Tud'] = np.divide(np.squeeze(FREQDOMAINVALUES[id]['INPUT'][IX]),FREQDOMAINVALUES[id]['DIS'][IX])
        FREQDOMAINVALUES[id]['Tyr'] = np.divide(np.squeeze(FREQDOMAINVALUES[id]['STATE'][IX]),FREQDOMAINVALUES[id]['REF'][IX])
        FREQDOMAINVALUES[id]['Tyd'] = np.divide(np.squeeze(FREQDOMAINVALUES[id]['STATE'][IX]),FREQDOMAINVALUES[id]['DIS'][IX])
        
        if (FREQDOMAINVALUES[id]['REF'] == 0).all():
            FREQDOMAINVALUES[id]['Tur'] = np.zeros(FREQDOMAINVALUES[id]['Tur'].shape, dtype=complex)
            FREQDOMAINVALUES[id]['Tyr'] = np.zeros(FREQDOMAINVALUES[id]['Tyr'].shape, dtype=complex)
            
        if (FREQDOMAINVALUES[id]['DIS'] == 0).all():
            FREQDOMAINVALUES[id]['Tud'] = np.zeros(FREQDOMAINVALUES[id]['Tud'].shape, dtype=complex)
            FREQDOMAINVALUES[id]['Tyd'] = np.zeros(FREQDOMAINVALUES[id]['Tyd'].shape, dtype=complex)

    return timedomainvalues, FREQDOMAINVALUES, xf, IX

def findControllers(oM, primes, fs, base_period, n_base_cycles, trials, sines_pairs):
    '''
    returns feedforward and feedback controllers for each pair of even/odd trials
    '''
    # find FFT
    timevalues, FREQVALUES, xf, IX = findFFT(primes, fs, base_period, n_base_cycles, trials)
    
    # get machine dynamics, given machine order
    s = 1.j*2*m.pi*xf
    zoM = 1./np.ones((len(s),))
    foM = 1./(s)
    soM = 1./(s**2+s)

    if oM == 0:
        M =  zoM[IX] # (n_freqs,)
    elif oM == 1:
        M = foM[IX]
    elif oM == 2:
        M = soM[IX]
    else:
        print('Machine order not recognized!')

    # initialize return variables
    timeerr = np.zeros((len(trials),))                            # (n_trials,)
    
    mean_freqerr = np.zeros((len(sines_pairs),), dtype=complex)   # (n_pairs,)
    freqerr = np.zeros((len(sines_pairs),len(IX)), dtype=complex) # (n_pairs, n_freqs)
    freqerr_sq = freqerr.copy()
    
    B = freqerr.copy()
    F = freqerr.copy()
    Tur_h = freqerr.copy()
    Tud_h = freqerr.copy()
    
    Tyr_freqerr = np.zeros((len(sines_pairs)),)                   # (n_pairs,)
    Tyd_freqerr = Tyr_freqerr.copy()

    # TIME DOMAIN
    for trial_id, trial in enumerate(trials):
        # calculate MSE in time domain --> (r-y)^2
        timeerr[trial_id] = np.mean( (timevalues[trial_id]['state'] - timevalues[trial_id]['ref'])**2 )

    # FREQ DOMAIN
    for pair_id, (id_a, id_b) in enumerate(sines_pairs):
        # find freqs that generated ref and dis for current and next trial (full set of stim freqs used every 2 trials)
        ref_ind = trials[id_a]['ref_ind'] # [1 3 5 7] or [0 2 4 6]
        dis_ind = trials[id_a]['dis_ind'] # [1 3 5 7] or [0 2 4 6], opposite of ref_ind
        next_ref_ind = trials[id_b]['ref_ind'] # ind of freqs used for ref in other trial in pair, opposite of current trial
        next_dis_ind = trials[id_b]['dis_ind'] # ind of freqs used for dis in other trial in pair, opposite of current trial

        # find transfer functions at stimulus freqs
        Tur = np.zeros((len(IX),), dtype=complex) # (n_freqs,)
        Tud = Tur.copy()
        Tur[ref_ind] = FREQVALUES[id_a]['Tur'][ref_ind]
        Tud[dis_ind] = FREQVALUES[id_a]['Tud'][dis_ind]
        Tur[next_ref_ind] = FREQVALUES[id_b]['Tur'][next_ref_ind]
        Tud[next_dis_ind] = FREQVALUES[id_b]['Tud'][next_dis_ind]

        Tyr = np.zeros((len(IX),), dtype=complex) # (n_freqs,)
        Tyd = Tyr.copy()
        Tyr[ref_ind] = FREQVALUES[id_a]['Tyr'][ref_ind]
        Tyd[dis_ind] = FREQVALUES[id_a]['Tyd'][dis_ind]
        Tyr[next_ref_ind] = FREQVALUES[id_b]['Tyr'][next_ref_ind]
        Tyd[next_dis_ind] = FREQVALUES[id_b]['Tyd'][next_dis_ind]

        # save freq-domain transfer functions
        Tur_h[pair_id,:] = Tur
        Tud_h[pair_id,:] = Tud
        
        # calculate feedforward and feedback controllers --> eq. 6a and 6b in Yamagami 2021 paper
        B[pair_id,:] = np.divide(-Tud,np.multiply(M,np.ones(Tud.shape,dtype=complex)+Tud))
        F[pair_id,:] = np.multiply(Tur,(1+0j)+np.multiply(B[pair_id,:],M)) - B[pair_id,:]

        # calculate MSE in freq domain --> |F-M^-1|
        freqerr[pair_id,:] = abs( F[pair_id,:] - np.divide(np.ones(M.shape),M) ) # error at each stim freq, NOT complex
        mean_freqerr[pair_id] = np.mean(freqerr[pair_id,:]) # avg error over stim freqs
        
        freqerr_sq[pair_id,:] = ( F[pair_id,:] - np.divide(np.ones(M.shape),M) )**2 # (F-M^-1)^2 at each stim freq, complex
            # (freqerr)^2  = abs(freqerr_sq)
            # (|F-M^-1|)^2 = |(F-M^-1)^2|
        
        # calculate other error in freq domain --> error in Tyr (ideal=1) and Tyd (ideal=0)
        Tyr_freqerr[pair_id] = abs(np.mean(Tyr - 1))
        Tyd_freqerr[pair_id] = abs(np.mean(Tyd - 0))
        
    return F, B, Tur_h, Tud_h, timeerr, freqerr, mean_freqerr, freqerr_sq, Tyr_freqerr, Tyd_freqerr, timevalues, FREQVALUES, xf, IX