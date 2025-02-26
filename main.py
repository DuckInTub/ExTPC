import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import scipy
import math

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


for frame_nr in range(total_nr_frames):
    pass

    # Calcualte the received power of the current frame.
    # Do this for each TPC method.

    # Calculate weather the current frame will be lost.
    # Do this for each TPC method.

    # Add the received power for the current frame
    # to the received_power data list of each TPC method.

    # Calculate the transmisson power of the next frame
    # Add that transmission power to the transmitted_power data list
    # Do this for each TPC method

# --- Section statistics ---

# --- Section graphing ---

