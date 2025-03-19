import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import scipy
import math

import scipy.io
from tpc_methods import *
#from test import simulate_path_loss
from util_functions import *

with open("out.csv", newline='') as dataset_csv:
    reader = csv.reader(dataset_csv)
    path_loss_list = list(reader)[0]
    path_loss_list = [float(PL) for PL in path_loss_list] # 60s at 1KHz -> 60_000 values
    # mat = scipy.io.loadmat("data/20080919-Male1_3kph.mat")

# path_loss_list = simulate_path_loss(1000, 60)

# Time = 60s
# Carrier frequency = 820MHz
# Sample rate = 1KHz
# Mean path loss = 55.4 dB
# Frame size = 800 bits
# Data rate = 151.8 kbps
# Transmission power 0 dB
frame_time = 5.2701 # ms
frame_interval = 200 # ms
total_nr_frames = math.ceil(1000 * 60 / frame_interval)

tx_power_min = -25 # dBm
rx_power_max = 0 # dBm
P_target = -85 # dBm
offset = 3 # dB

TPC_methods : dict[str, TPC_method_interface] = {
    "Constant": Constant(-10),
    "Xiao_aggressive": Xiao_aggressive(),
    "Gao": Gao(),
    "Sodhro": Sodhro(),
    # "Optimal": Optimal(), # Don't know if we should have this one?
}

# For all the TPC methods calculate the supposed tx_power for
# the first frame. We use the average path loss of the dataset.
path_loss_avg = np.average(path_loss_list)
first_tx_power = variables.rx_power_target - path_loss_avg
for method in TPC_methods.values():
    method.current_rx_power = variables.rx_power_target
    method.current_tx_power = first_tx_power
    method.update_internal()

for frame_nr in range(total_nr_frames):

    for method in TPC_methods.values():
        # Calcualte the received power of the current frame.
        # Line 333 in Matlab
        start_of_frame = frame_nr*frame_interval
        frame_path_losses = path_loss_list[start_of_frame:start_of_frame+math.floor(frame_time)]
        method.current_rx_power = np.average([method.current_tx_power + path_loss for path_loss in frame_path_losses])

        # Calculate weather the current frame will be lost.
        if method.current_rx_power < P_target - offset:
            method.lost_frames.append(frame_nr)

        # Add the received power for the current frame
        # to the received_power data list of each TPC method.
        method.rx_powers.append(method.current_rx_power)

        # According to the papers different methods use different ways to calculate the running average.
        # This method computes such averages and other internal state variables. 
        method.update_internal()

        # Calculate the transmisson power of the next frame
        # Add that transmission power to the tx_power data list history
        method.current_tx_power = method.next_transmitt_power(P_target, offset)
        method.tx_powers.append(method.current_tx_power)

# --- Section statistics ---
for name, method in TPC_methods.items():
    lost_frames = method.lost_frames
    avg_tx_power = np.average(method.tx_powers)
    pack_loss_ratio = 100*len(lost_frames) / total_nr_frames
    power_consumed = sum(map(tx_power_to_mW, method.tx_powers))
    print(f"{name}: Packet loss {pack_loss_ratio:.2f}% with avg_tx_power: {avg_tx_power:.2f}, and power consumption {power_consumed:.2f}")

# # --- Section graphing ---
plt.plot(TPC_methods["Constant"].rx_powers, label="Constant")
plt.plot(TPC_methods["Xiao_aggressive"].rx_powers, label="Xiao_aggressive")
plt.plot(TPC_methods["Gao"].rx_powers, label="Gao")
plt.plot(TPC_methods["Sodhro"].rx_powers, label="Sodhro")
plt.legend()
plt.grid(True)
plt.show()

