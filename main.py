import numpy as np
import matplotlib.pyplot as plt
from util_functions import gen_sig_wmban, wgn, jakesm_siso, manipulate_fade_depths
import time

#----------------------------------------------------------------------------
# Random Seed Initialization
#----------------------------------------------------------------------------
# In MATLAB: rand('state', sum(100*clock));
# Here we seed numpy’s random generator using a value based on the current local time.
seed_value = int(np.sum(100 * np.array(time.localtime()[:6])))
np.random.seed(seed_value)

#----------------------------------------------------------------------------
# Simulation Parameters
#----------------------------------------------------------------------------
# Time (s) over which signal is generated (choose one)
# sim_time = 50
sim_time = 60

# Carrier frequency (MHz)
car_frequency = 420
# For example, you might also use: car_frequency = 2400

# Sample rate (kHz)
sample_rate = 1

# Relative body movement velocity (km/h)
vel = 1.5 + 4 * np.random.rand()

# Mean path loss (dB)
a = 60.2  # Scale parameter
b = 6.6   # Shape parameter
mean_path_loss = a * ((-np.log(np.random.rand())) ** (1 / b))

#----------------------------------------------------------------------------
# Simulation Start Setup
#----------------------------------------------------------------------------
snap_shot_num = 13  # Number of Monte Carlo Simulations

# Lists to hold simulation results
P_received_gao       = []  # Received power (dBm) for each data frame (our scheme)
P_transmitted_gao    = []  # Transmit power (dBm) for each data frame (our scheme)
P_received_xiao   = []  # Received power (dBm) for Reference Scheme
P_transmitted_xiao= []  # Transmit power (dBm) for Reference Scheme
P_received_const = []  # Received power (dBm) for Constant Power
P_transmitted_const = []  # Transmit power (dBm) for Constant Power
P_received_sodhro   = []  # Received power (dBm) for Modified/New scheme
P_transmitted_sodhro= []  # Transmit power (dBm) for Modified/New scheme
P_received_target= []  # Target RSSI (dBm)

# Packet loss counters
frame_loss_num_gao      = 0
frame_loss_num_xiao  = 0
frame_loss_num_const= 0
frame_loss_num_sodhro  = 0

# Simulation parameters for data transmission
frame_size = 100 * 8      # Data frame size in bits
data_rate = 151.8         # Data rate (kbps)
# Convert frame duration to an integer number of samples (in ms)
frame_time = int(np.round(frame_size / data_rate))  # Duration of a frame (ms)

frame_interval = 200      # Frame interval (ms)
frame_num = int(np.floor(1000 * sim_time * snap_shot_num / frame_interval))
# frame_num is the number of data frames during the simulation

p_min = -25    # Minimum transmission power (dBm)
p_max = -10    # Maximum transmission power (dBm)
p_target = -85 # Target received power (dBm)
offset = 3     # Offset (dB)
filter_coeff = 0.8  # Filter coefficient for channel smoothing

# Noise parameters
noise_psd = -174      # White noise power spectral density (dBm/Hz)
noise_figure = 5      # Receiver noise figure (dB)
bandwidth = 320       # Channel bandwidth (kHz)

# Power control step sizes (dB)
PC_step = np.array([-3, -2, -1, 0, 1, 2, 3, 4])

#----------------------------------------------------------------------------
# Monte Carlo Simulation Loop
#----------------------------------------------------------------------------
for ss in range(snap_shot_num):
    # Initialize averages for this simulation snapshot
    p_received_average_gao = 0
    p_current_gao = 0

    p_received_average_xiao = 0
    p_current_xiao = 0

    p_received_average_const = 0
    p_current_const = 0

    p_received_average_sodhro = 0
    p_current_sodhro = 0

    # Generate signal over the entire simulation time for this snapshot.
    # Note: gen_sig_wmban is assumed to be defined elsewhere.
    signal = gen_sig_wmban(vel, car_frequency, sample_rate, sim_time * snap_shot_num, mean_path_loss)

    #--------------------------------------------------------------------------
    # Simulate Attenuation (e.g. due to signal blocking)
    #--------------------------------------------------------------------------
    # In MATLAB, the second half of the signal is attenuated by -10 dB.
    signal_attenuation = 10 ** (-10 / 10)  # converts -10 dB to linear scale (0.1)
    half_idx = int(0.5 * len(signal))
    signal[half_idx:] *= signal_attenuation

    #--------------------------------------------------------------------------
    # Generate and Add Gaussian White Noise
    #--------------------------------------------------------------------------
    noise_power = 10 ** (((noise_psd + noise_figure) / 10) - 3) * bandwidth * 1000  # in Watts
    # Convert noise power to dBW (for use with our wgn function)
    noise_power_dBW = 10 * np.log10(noise_power / 1000)
    # Instead of a loop, generate noise for all samples at once:
    noise = wgn(1, len(signal), noise_power_dBW).flatten()
    signal = signal + noise

    #--------------------------------------------------------------------------
    # Initialize Average Received Power
    #--------------------------------------------------------------------------
    # The initial average is set to p_target (in dBm) converted to linear scale.
    p_received_average = 10 ** (p_target / 10)

    #--------------------------------------------------------------------------
    # Calculate Average Channel Gain for the First Frame
    #--------------------------------------------------------------------------
    # (i.e. the average over the first frame's samples)
    channel_frame_1 = np.mean(signal[:frame_time])

    # Compute the transmission power for the first frame
    Pt_gao = p_received_average / channel_frame_1
    Pt_xiao = p_received_average / channel_frame_1
    Pt_const = p_received_average / channel_frame_1
    Pt_sodhro = p_received_average / channel_frame_1

    # Convert Pt values from linear to dBm
    Pt_gao = 10 * np.log10(Pt_gao)
    Pt_xiao = 10 * np.log10(Pt_xiao)
    # For constant power, the MATLAB code sets a fixed value:
    Pt_const = -15
    Pt_sodhro = 10 * np.log10(Pt_sodhro)

    # Clip the transmission power to the allowed range
    Pt_gao = np.clip(Pt_gao, p_min, p_max)
    Pt_xiao = np.clip(Pt_xiao, p_min, p_max)
    Pt_const = np.clip(Pt_const, p_min, p_max)
    Pt_sodhro = min(Pt_sodhro, p_max)

    # Save the first (and second) frame’s transmission and received power values
    P_transmitted_gao.append(Pt_gao)
    P_transmitted_xiao.append(Pt_xiao)
    P_transmitted_const.append(Pt_const)
    P_transmitted_sodhro.append(Pt_sodhro)

    P_received_gao.append(p_target)
    P_received_xiao.append(p_target)
    P_received_const.append(p_target)
    P_received_sodhro.append(p_target)
    P_received_target.append(p_target)

    # The second frame’s transmit power is the same as the first frame’s.
    P_transmitted_gao.append(Pt_gao)
    P_transmitted_xiao.append(Pt_xiao)
    P_transmitted_const.append(Pt_const)
    P_transmitted_sodhro.append(Pt_sodhro)

    #--------------------------------------------------------------------------
    # Process Data Frames (from the second frame onward)
    #--------------------------------------------------------------------------
    # Note: MATLAB frames are indexed starting at 1. Here we use 0-based indexing.
    for fn in range(1, frame_num):
        # Compute average received power (in linear scale) for current frame
        start_idx = fn * frame_interval
        end_idx = start_idx + frame_time

        # Our scheme: average received power of current frame
        p_current_gao = np.mean(10 ** (Pt_gao / 10) * signal[start_idx:end_idx])
        # Reference scheme
        p_current_xiao = np.mean(10 ** (Pt_xiao / 10) * signal[start_idx:end_idx])
        # Constant power scheme
        p_current_const = np.mean(10 ** (Pt_const / 10) * signal[start_idx:end_idx])
        # Modified/New scheme uses the current frame and also the previous frame:
        p_current_sodhro = np.mean(10 ** (Pt_sodhro / 10) * signal[start_idx:end_idx])
        # p_lowest_new: average from the previous frame
        prev_start = (fn - 1) * frame_interval
        prev_end = prev_start + frame_time
        p_lowest_sodhro = np.mean(10 ** (Pt_sodhro / 10) * signal[prev_start:prev_end])

        # Determine if a packet (data frame) is lost (i.e. received power too low)
        if 10 * np.log10(p_current_gao) < p_target - offset:
            frame_loss_num_gao += 1
        if 10 * np.log10(p_current_xiao) < p_target - offset:
            frame_loss_num_xiao += 1
        if 10 * np.log10(p_current_const) < p_target - offset:
            frame_loss_num_const += 1
        if 10 * np.log10(p_current_sodhro) < p_target - offset:
            frame_loss_num_sodhro += 1
        if 10 * np.log10(p_lowest_sodhro) < p_target - offset:
            frame_loss_num_sodhro += 1

        # Save received power (in dBm) for this frame
        P_received_gao.append(10 * np.log10(p_current_gao))
        P_received_xiao.append(10 * np.log10(p_current_xiao))
        P_received_const.append(10 * np.log10(p_current_const))
        P_received_sodhro.append(10 * np.log10(p_current_sodhro))
        P_received_target.append(p_target)

        # Update the averaged received power (smoothing) for each scheme
        p_received_average_gao = (1 - filter_coeff) * p_received_average_gao + filter_coeff * p_current_gao
        p_received_average_xiao = (1 - filter_coeff) * p_received_average_xiao + filter_coeff * p_current_xiao
        p_received_average_const = (1 - filter_coeff) * p_received_average_const + filter_coeff * p_current_const
        p_received_average_sodhro = (1 - filter_coeff) * p_received_average_sodhro + p_lowest_sodhro

        #----------------------------------------------------------------------
        # Adjust Transmission Power for Next Frame (if not the last frame)
        #----------------------------------------------------------------------

        # --- Our Scheme ---
        if fn != frame_num - 1:
            current_avg_dB = 10 * np.log10(p_received_average)
            # In MATLAB the same expression is used in both the "greater than" and
            # "less than" conditions. We replicate that logic here.
            if (current_avg_dB > p_target + offset) or (current_avg_dB < p_target - offset):
                diff = p_target - current_avg_dB
                possible = PC_step[PC_step > diff]
                if possible.size > 0:
                    p_delta_gao = possible.min()
                else:
                    p_delta_gao = PC_step.max()
            else:
                p_delta_gao = 0

            Pt_gao = Pt_gao + p_delta_gao
            Pt_gao = np.clip(Pt_gao, p_min, p_max)
            P_transmitted_gao.append(Pt_gao)

        # --- Reference Scheme ---
        if fn != frame_num - 1:
            current_avg_xiao = 10 * np.log10(p_received_average_xiao)
            if current_avg_xiao > p_target + offset:
                p_delta_xiao = -1
            elif current_avg_xiao < p_target - offset:
                p_delta_xiao = 3
            else:
                p_delta_xiao = 0
            Pt_xiao = Pt_xiao + p_delta_xiao
            Pt_xiao = np.clip(Pt_xiao, p_min, p_max)
            P_transmitted_xiao.append(Pt_xiao)

        # --- Constant Power Scheme ---
        if fn != frame_num - 1:
            p_delta_const = 0  # remains unchanged
            Pt_const = Pt_const + p_delta_const
            P_transmitted_const.append(Pt_const)

        # --- Modified/New Scheme ---
        if fn != frame_num - 1:
            current_avg_sodhro = 10 * np.log10(p_received_average_sodhro)
            if current_avg_sodhro > p_target + offset:
                p_delta_sodhro = -1
            elif current_avg_sodhro < p_target - offset:
                p_delta_sodhro = 2
            else:
                p_delta_sodhro = 0
            Pt_sodhro = Pt_sodhro + p_delta_sodhro
            Pt_sodhro = np.clip(Pt_sodhro, p_min, p_max)
            P_transmitted_sodhro.append(Pt_sodhro)

#----------------------------------------------------------------------------
# Data Statistics Calculation
#----------------------------------------------------------------------------
# Average Transmission Power (linear scale)
pt_average_gao = np.mean([10 ** (pt / 10) for pt in P_transmitted_gao])
pt_average_xiao = np.mean([10 ** (pt / 10) for pt in P_transmitted_xiao])
pt_average_const = np.mean([10 ** (pt / 10) for pt in P_transmitted_const])
pt_average_sodhro = np.mean([10 ** (pt / 10) for pt in P_transmitted_sodhro])

# Standard Deviation of Received Power (using deviation from target)
pt_dev_gao = np.sqrt(np.mean([(pr - p_target) ** 2 for pr in P_received_gao]))
pt_dev_xiao = np.sqrt(np.mean([(pr - p_target) ** 2 for pr in P_received_xiao]))
pt_dev_const = np.sqrt(np.mean([(pr - p_target) ** 2 for pr in P_received_const]))
pt_dev_sodhro = np.sqrt(np.mean([(pr - p_target) ** 2 for pr in P_received_sodhro]))

#----------------------------------------------------------------------------
# Plotting Results
#----------------------------------------------------------------------------
# Time axis (in seconds)
total_frames = frame_num * snap_shot_num
time_axis = np.arange(1, total_frames + 1) * frame_interval / 1000  # converting ms to s

plt.figure(1)
plt.plot(time_axis, P_received_const, 'g', label='Constant')
plt.plot(time_axis, P_received_xiao, 'r', label='Xiao Method')
plt.plot(time_axis, P_received_gao, 'b:', label='Gao Method')
plt.plot(time_axis, P_received_sodhro, 'k-', label='Sodhro Method')
plt.plot(time_axis, P_received_target, 'c.', label='RSSI Target')
plt.xlabel('Time (s)')
plt.ylabel('RSSI (dBm)')
plt.axis([0, 60, -100, -50])
plt.legend()

plt.figure(2)
plt.plot(time_axis, P_transmitted_const, 'g.', label='Constant')
plt.plot(time_axis, P_transmitted_xiao, 'r-', label='Xiao Method')
plt.plot(time_axis, P_transmitted_gao, 'b:', label='Gao Method')
plt.plot(time_axis, P_transmitted_sodhro, 'k-', label='Sodhro Method')
plt.xlabel('Time (s)')
plt.ylabel('Transmit Power (dBm)')
plt.axis([0, 60, -30, 5])
plt.legend()

plt.show()

#----------------------------------------------------------------------------
# Display Simulation Data
#----------------------------------------------------------------------------
print("Frame Loss Counts:")
print("Original Scheme:", frame_loss_num_gao)
print("Reference Scheme:", frame_loss_num_xiao)
print("Constant Power:", frame_loss_num_const)
print("Modified Scheme:", frame_loss_num_sodhro)
print("")
print("Average Transmission Power (linear scale):")
print("Original Scheme:", pt_average_gao)
print("Reference Scheme:", pt_average_xiao)
print("Constant Power:", pt_average_const)
print("Modified Scheme:", pt_average_sodhro)
print("")
print("Standard Deviation of Received Power (dB):")
print("Original Scheme:", pt_dev_gao)
print("Reference Scheme:", pt_dev_xiao)
print("Constant Power:", pt_dev_const)
print("Modified Scheme:", pt_dev_sodhro)

# End of simulation script
a = 0  # (Dummy assignment as in the MATLAB code)