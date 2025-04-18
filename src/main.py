import numpy as np
import matplotlib.pyplot as plt
import math
from tpc_methods import *
from util_functions import *
from pathlib import Path
import pickle
import time

# Load data from .mat file
# path = Path("..") / "data" / "Male1_3kph.mat"
# path_loss_list = load_mat_file(path)

# Load data from pickle
path = Path("..") / "data" / "data.pkl"
with open(path, "rb") as file:
    data_dict = pickle.load(file)
    most_data_keys = ["pl_LeftWrist_RightHip", "pl_RightWrist_RightHip", "pl_LeftAnkle_RightHip"]
    indx = 0
    path_loss_list = []
    for person, data in data_dict[most_data_keys[indx]]["6kph"].items():
        path_loss_list += data
    del data_dict

# Simulate data
start_time = time.perf_counter()
# path_loss_list = simulate_path_loss(1000, 60)

# Parameters
frame_time = 5.2701  # ms
frame_interval = 200  # ms
total_nr_frames = len(path_loss_list) // frame_interval

tx_power_min = -25  # dBm
rx_power_max = 0  # dBm
P_target = -85  # dBm
packet_loss_RSSI = -88 # Cite the standard

frame_processing_time = 3  # ms
ack_frame_time = 0.2  # ms

frame_path_losses = extract_frames(path_loss_list, frame_interval, frame_time)
del path_loss_list
print(f"Signal loaded. {time.perf_counter() - start_time:2f}s")
print(f"Total number of frames: {total_nr_frames}")

TPC_methods: dict[str, TPCMethodInterface] = {
    "Optimal": Optimal(frame_path_losses, packet_loss_RSSI, total_nr_frames),
    "Constant": Constant(total_nr_frames, -10),
    "Naive": Naive(total_nr_frames),
    "Xiao_aggr_2008": Xiao_aggressive_2008(total_nr_frames),
    "Xiao_cons_2008": Xiao_conservative_2008(total_nr_frames),
    "Xiao_aggr_2009": Xiao_2009(total_nr_frames, 0.2, 0.8),
    "Xiao_bal_2009": Xiao_2009(total_nr_frames, 0.8, 0.8),
    "Xiao_cons_2009": Xiao_2009(total_nr_frames, 0.8, 0.2),
    "Gao": Gao(total_nr_frames),
    "Sodhro": Sodhro(total_nr_frames),
    "Guo": Guo(total_nr_frames),
    "Smith": Smith_2011(total_nr_frames, 3),
}

# Main simulation loop
for frame_nr, frame_path_loss in enumerate(frame_path_losses):
    for name, method in TPC_methods.items():
        method.update_current_rx(method.current_tx_power + frame_path_loss)

        # Check if packet is lost in the current frame
        packet_lost = method.current_rx_power < packet_loss_RSSI

        method.track_stats(
            frame_nr, packet_lost, frame_interval,
            frame_processing_time, ack_frame_time
        )

        # Give TPC algorithms feedback information and let them update TX power
        if name.startswith("Xiao"):
            method.update_transmission_power(-82.5, -85, -80)
        elif name == "Gao":
            method.update_transmission_power(-82.5, -85, -80)
        elif name == "Sodhro":
            method.update_transmission_power(-85, -88, -82.3)
        else:
            method.update_transmission_power(P_target, -85, -80)

print(f"Main simulation loop completed. {time.perf_counter() - start_time:2f}s")

print(f"Number of samples {total_nr_frames}")
print(TPCMethodInterface.get_stats_header())

for name, method in TPC_methods.items():
    print(method.calculate_stats(name, frame_time, total_nr_frames))

# # Plot and summary statistics
# plt.figure(figsize=(16, 9))
# for name, method in TPC_methods.items():
#     plt.plot(method.rx_powers, label=name)
# # Plot settings
# plt.xlim(0, total_nr_frames)
# plt.ylim(-120, -60)
# plt.title("Received power (dBm) over time", fontsize=16, fontweight="bold")
# plt.xlabel("Frame nr", fontsize=14, fontweight="bold")
# plt.ylabel("P_rx (dBm)", fontsize=14, fontweight="bold")
# plt.legend()
# plt.grid(True)