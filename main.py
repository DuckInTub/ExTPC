import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import scipy
import math

import scipy.io
from tpc_methods import *

with open("data_csvs/out.csv", newline='') as dataset_csv:
    reader = csv.reader(dataset_csv)
    path_loss_list = list(reader)[0]
    path_loss_list = [float(PL) for PL in path_loss_list] # 60s at 1KHz -> 60_000 values
    print(len(path_loss_list))
    # mat = scipy.io.loadmat("data/20080919-Male1_3kph.mat")

# Time = 60s
# Carrier frequency = 820MHz
# Sample rate = 1KHz
# Mean path loss = 55.4 dB
# Frame size = 800 bits
# Data rate = 151.8 kbps
# Transmission power 0 dB
frame_time = 5.2701 # ms
frame_interval = 100 # ms
total_nr_frames = math.ceil(1000 * 60 / frame_interval)

tx_power_min = -25 # dBm
rx_power_max = 0 # dBm
P_target = -85 # dBm
offset = 3 # dB
avg_weight = 0.8 # alpha in Xiaos paper

TPC_methods : dict[str, TPC_method_interface] = {
    "Constant": Constant(-10),
    "Xiao_aggressive": Xiao_aggressive(),
    "Gao": Gao(),
    # "Sodhro": Sodhro(),
    # "Optimal": Optimal(), # Don't know if we should have this one?
}

for frame_nr in range(total_nr_frames):


    for method in TPC_methods.values():

        # TODO Check the initialization state of the simulation.
        # Reference matlab. Sodhro needs some stuff

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

        # Calculate running average using exponential averaging for rx_power and update it
        # This is R̅ in Xiao's paper under aggressive TCP method.
        method.exp_avg_rx_power = (1 - avg_weight)*method.exp_avg_rx_power + avg_weight*method.current_rx_power
        # TODO According to the papers different methods use different ways to calculate the running average

        # Calculate the transmisson power of the next frame
        # Add that transmission power to the tx_power data list history
        method.current_tx_power = method.next_transmitt_power(P_target, offset)
        method.tx_powers.append(method.current_tx_power)

# --- Section statistics ---
print(TPC_methods["Gao"].rx_powers)
print(TPC_methods["Gao"].lost_frames)
print(len(TPC_methods["Gao"].lost_frames))
# --- Section graphing ---

