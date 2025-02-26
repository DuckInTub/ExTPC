import numpy as np

def manipulate_fade_depths(initial_signal, mean_path_loss):
    quantised = (initial_signal / np.mean(initial_signal)) < 1
    crossings_up = np.where(np.diff(quantised) == -1)[0]
    crossings_down = np.where(np.diff(quantised) == 1)[0]

    if len(crossings_up) > 1 and len(crossings_down) > 1:
        if crossings_up[0] < crossings_down[0]:
            crossings_up = crossings_up[1:]
        if crossings_down[-1] > crossings_up[-1]:
            crossings_down = crossings_down[:-1]

    nfades = len(crossings_up)

    # Uniformly distributed random numbers on (0, 1)
    randnum = np.random.rand(nfades)  # generates an array of shape (nfades,)

    a = 8.4  # Weibull shape parameter
    b = 0.77  # Weibull scale parameter

    # Generate Weibull-distributed random numbers:
    # In MATLAB: wp = a*(-log(randnum)).^(1/b);
    wp = a * (-np.log(randnum))**(1/b)

    # Place Weibull random numbers around mean path loss
    fade_mags = -mean_path_loss - wp

    # Exclude unrealistic fade magnitudes.
    # (Assuming fade_mags is a NumPy array and mean_path_loss is defined)
    fade_mags = fade_mags[fade_mags > -mean_path_loss - 73]

    # Sort fade_mags in descending order.
    # For a 1D array, np.sort sorts in ascending order, so reverse it.
    fade_mags = np.sort(fade_mags)[::-1]
    # If fade_mags were a 2D array and you want to sort along rows (axis=1) in descending order, you could do:
    # fade_mags = np.sort(fade_mags, axis=1)[:, ::-1]

    # Convert the initial signal to dB.
    initial_signal_dB = 10 * np.log10(initial_signal)

    # Find the index of the start of the first fade.
    # MATLAB's crossings_down(1) corresponds to crossings_down[0] in Python.
    index_start_fade = crossings_down[0]
    # If the initial signal starts in a fade, then we create an empty output signal.

    # Otherwise, we insert the portion of the initial signal (before any fades occur).
    if index_start_fade > 0:
        signal_dB = initial_signal_dB[:index_start_fade]
    else:
        signal_dB = np.array([])

    # Find how many crossings above the threshold there are.
    total_crossings = len(crossings_up)

    # Initialize a counter for how many signal portions have been used so far.
    kt = 0

    # Initialize signal_portion to store the signal segments.
    signal_portion = []

    #--------------------------------------------------------------------------
    # Fade manipulation:
    # Apply the previously described method of keeping/clipping/removing faded
    # sections of the initial signal.
    #--------------------------------------------------------------------------

    # Iterate over each fade & non-fade portion of the initial signal
    for j in range(total_crossings):

        # Find the index of the crossing (above the mean) in the initial
        # signal. This index marks the end of the fade.
        index_end_fade = crossings_up[j] - 1

        # Find the portion of signal that corresponds to this fade
        current_fade = initial_signal_dB[index_start_fade : index_end_fade + 1]

        # Find the magnitude/depth of the current fade
        current_fade_magnitude = np.min(current_fade)

        # Compare the current fade magnitude to the sorted vector of desired
        # fade magnitudes. We find the index of the closest desired fade depth
        # whose fade is of greater magnitude than the current fade magnitude.
        indices = np.where(fade_mags > current_fade_magnitude)[0]
        if indices.size == 0:
            ks = None
        else:
            ks = np.max(indices)

        # If such an index exists we manipulate the initial faded signal
        # portion appropriately
        if ks is None:
            # No matching fade depth was found, do not use this portion of the
            # signal.
            ts = 0  # Flag that the current signal portion was not used

        else:
            fm = fade_mags[ks]  # Find the value of given fade depth

            # Remove this fade magnitude from the vector of possible magnitudes
            # (we only use each fade magnitude once).
            fade_mags = fade_mags[fade_mags != fm]

            # Remove the parts of the current signal portion which have too
            # much attenuation.
            current_fade = current_fade[current_fade > fm]

            # Use the current signal portion?
            if current_fade.size > 0:
                kt = kt + 1  # Increase counter of signal portions used
                ts = 1      # Flag that the current signal portion was used
            else:
                # The current signal portion is empty, can't use it.
                ts = 0      # Flag that the current signal portion was not used

        #----------------------------------------------------------------------
        # Collect all the fades together
        #----------------------------------------------------------------------

        # If we have not iterated through all of the fades then insert the
        # current signal portion (fade + non-fade) into the cell array of signal
        # portions, as appropriate. This cell array will be used to generate
        # the final output signal.
        if j < total_crossings - 1:
            # Set index for start of next fade
            index_start_fade = crossings_down[j+1]

            # If current signal portion was used...
            if ts == 1:
                # Take the current fade as well as the next contiguous non-fade
                # portion of the signal and insert it into the cell array.
                # MATLAB: signal_portion{kt} = [current_fade, initial_signal_dB((index_end_fade+1):(index_start_fade-1))];
                signal_portion.append(np.concatenate((current_fade, initial_signal_dB[index_end_fade+1 : index_start_fade])))
        else:
            # There is no non-fade portion following the current fade, so we
            # just add the fade to the cell array.
            if ts == 1:
                signal_portion.append(current_fade)

        # If all the generated fade depths have been used, exit the 'for' loop
        if fade_mags.size == 0:
            break
    #--------------------------------------------------------------------------
    # Jumble the signal
    #--------------------------------------------------------------------------

    # We finally jumble up each signal portion that we stored in the cell array
    # (note: not jumbling within portions). This is necessary due to the
    # unnatural ordering of fades that occurs from our method of matching fade
    # depths.

    Rp = np.random.permutation(kt)

    for kl in range(kt):
        signal_dB = np.concatenate((signal_dB, signal_portion[Rp[kl]]))
    
    return signal_dB


def gen_sig_wmban(vel, car_frequency, sample_rate, time, mean_path_loss):
    """
    Generates a received power profile vector for an on-body mobile wireless BAN.
    
    Parameters:
      vel            - Speed of movement (km/h)
      car_frequency  - Carrier frequency (MHz)
      sample_rate    - Sample rate (kHz)
      time           - Duration over which the signal is generated (s)
      mean_path_loss - Expected mean path loss for the channel (dB)
      
    Returns:
      signal - A power profile (absolute magnitude, not in dB) describing receive power 
               as a fraction of transmit power.
    """
    # Check that parameters lie within the acceptable ranges.
    if car_frequency < 400 or car_frequency > 2500:
        print('Carrier frequency should be in range 400 -- 2500 MHz')
        return np.array([])
    if sample_rate < 0.75 or sample_rate > 15:
        print('Sample rate should be in range 0.75 -- 15 kHz')
        return np.array([])
    if vel < 1.5 or vel > 20:
        print('Velocity should be in range 1.5 -- 20 km/h')
        return np.array([])

    #--------------------------------------------------------------------------
    # Set parameters used to generate signal.
    #--------------------------------------------------------------------------
    # Maximal Doppler frequency shift (Hz)
    fd = (vel / 3.6) * (car_frequency * 1e6) / 3e8
    # Sample rate conversion to hertz (Hz)
    fs = sample_rate * 1e3
    # Normalized Doppler spread (fading parameter)
    p = fd / fs
    # Total number of samples over which the final signal is generated.
    nsamp = int(np.round(time * fs))

    #--------------------------------------------------------------------------
    # Increase number of samples if required
    #--------------------------------------------------------------------------
    time_change = False
    nsamp1 = nsamp  # Keep original number of samples for possible later truncation.
    if car_frequency < 800 and time < 35:
        time = 35
        nsamp1 = nsamp
        nsamp = int(np.round(time * fs))
        time_change = True
    if time < 20:
        time = 20
        nsamp1 = nsamp
        nsamp = int(np.round(time * fs))
        time_change = True

    #--------------------------------------------------------------------------
    # Signal generation loop
    #--------------------------------------------------------------------------
    enough_samples = False
    while not enough_samples:
        # Generate a vector of Weibull distributed random numbers.
        rand_length = int(np.round(1.6 * nsamp))
        randnum = np.random.rand(rand_length)  # Uniformly distributed numbers in (0, 1)
        
        # Weibull parameters from measurement data.
        a = 0.9926  # shape parameter
        b = 0.9832  # scale parameter
        
        # Generate Weibull distributed random numbers.
        Al = a * np.power(-np.log(randnum), 1.0 / b)
        AdB = 10 * np.log10(Al)  # Convert to dB
        
        # Discard values outside the realistic range.
        mask = (AdB > -73) & (AdB < 21)
        AdB = AdB[mask]
        
        # Generate a Rayleigh fading power profile using Jakes' model.
        # (jakesm_siso should return a complex-valued array.)
        Rayleigh_power = np.abs(jakesm_siso(len(AdB), p))**2
        
        # Sort Rayleigh_power and get the sorting indices.
        I = np.argsort(Rayleigh_power)
        
        # Generate the unordered received signal power in dB.
        P = AdB - mean_path_loss
        A_f = 10 ** (P / 10)  # Convert from dB to absolute power
        
        # Order A_f according to the Rayleigh fading ordering.
        sorted_Af = np.sort(A_f)
        initial_signal = np.empty_like(A_f)
        initial_signal[I] = sorted_Af
        
        # Manipulate fade depths to obtain the desired distribution.
        signal_dB = manipulate_fade_depths(initial_signal, mean_path_loss)
        
        if len(signal_dB) >= nsamp:
            enough_samples = True

    # Truncate the generated signal to the desired number of samples and convert from dB.
    signal = 10 ** (signal_dB[:nsamp] / 10)
    if time_change:
        signal = signal[:nsamp1]

    # Readjust the signal so its mean matches the specified mean path loss.
    signal_mean_dB_non_fixed = 10 * np.log10(np.mean(signal))
    signal_dB_fin = 10 * np.log10(signal) - (mean_path_loss + signal_mean_dB_non_fixed)
    signal = 10 ** (signal_dB_fin / 10)
    
    return signal

def jakesm_siso(N1, p):
    """
    Generates Rayleigh fading for a single-input single-output system 
    according to Jakes' model [Jakes74].

    Parameters:
      N1 : int
           Desired number of samples for the channel response.
      p  : float
           Doppler fading parameter (normalized Doppler spread by sample rate).

    Returns:
      Hs : ndarray (complex)
           Complex fade amplitudes (length N1).
    """
    Ns = 50  # Number of scatterers
    c = 1.0 / np.sqrt(Ns)  # Normalization factor

    # Generate scatterer phase shifts; one row of random numbers repeated for N1 rows.
    ts = np.tile(2 * np.pi * np.random.rand(1, Ns), (N1, 1))
    
    # Create a column vector for sample indices (MATLAB uses 1:N1)
    n = np.arange(1, N1 + 1).reshape((N1, 1))
    
    # Generate cosine terms with random phases, repeated across rows.
    cos_term = np.cos(2 * np.pi * np.random.rand(1, Ns))
    
    # Compute the frequency term using broadcasting.
    ft = 2 * np.pi * p * n * cos_term

    # Sum over the scatterers to produce the complex fade amplitudes.
    Hs = np.sum(c * np.exp(-1j * (ft + ts)), axis=1)
    
    return Hs

def wgn(m, n, noise_power_dBW):
    """
    Generate white Gaussian noise.
    noise_power_dBW is in dBW (i.e. dB relative to 1 W).
    Returns an (m x n) array of noise samples.
    """
    noise_power_linear = 10 ** (noise_power_dBW / 10)  # linear scale (W)
    return np.random.randn(m, n) * np.sqrt(noise_power_linear)