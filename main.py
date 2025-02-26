import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import scipy
import math
from tpc_methods import *

with open("data_csvs/out.csv", newline='') as dataset_csv:
    reader = csv.reader(dataset_csv)
    path_loss_list = list(reader)[0]
    path_loss_list = [float(PL) for PL in path_loss_list]

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

P_min = -25 # dBm
P_max = 0 # dBm
P_target = -85 # dBm
offset = 3 # dB

print(path_loss_list)
print(np.average(path_loss_list))

TCP_methods : list[TPC_method_interface] = [
    Constant(),
    Xiao(),
    Gao(),
    Sodhro(),
    Optimal(), # Don't know if we should have this one?
] 


for frame_nr in range(total_nr_frames):


    for method in TCP_methods:

        # Calcualte the received power of the current frame.
        # Line 333 in Matlab
        start_of_frame = frame_nr*frame_interval
        frame_path_losses = path_loss_list[start_of_frame:start_of_frame+5]
        method.current_rx_power = np.average([method.current_tx_power - path_loss for path_loss in frame_path_losses])

        # Calculate weather the current frame will be lost.
        if method.current_rx_power < P_target - offset:
            method.lost_frames.append(frame_nr)

        # Add the received power for the current frame
        # to the received_power data list of each TPC method.
        method.rx_powers.append(method.current_rx_power)

        # TODO Calculate running average for rx_power and update it

        # TODO Calculate the transmisson power of the next frame
        # Add that transmission power to the transmitted_power data list
        method.next_transmitt_power()

# --- Section statistics ---

# --- Section graphing ---

