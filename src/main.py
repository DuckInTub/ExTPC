import numpy as np
import matplotlib.pyplot as plt
import math
from tpc_methods import *
from util_functions import *
from pathlib import Path
import pickle
import time

# Load data from .mat file
path = Path("..") / "data" / "Male1_3kph.mat"
path_loss_list = load_mat_file(path)

# Load data from pickle
# path = Path("..") / "data" / "data.pkl"
# with open(path, "rb") as file:
#     data_dict = pickle.load(file)
#     path_loss_list = navigate_dict(data_dict)
#     del data_dict

# Simulate data
start_time = time.perf_counter()
# path_loss_list = simulate_path_loss(1000, 600)

# Parameters
frame_time = 5.2701  # ms
frame_interval = 200  # ms
total_nr_frames = len(path_loss_list) // frame_interval

tx_power_min = -25  # dBm
rx_power_max = 0  # dBm
P_target = -85  # dBm
packet_loss_RSSI = -88 # Cite the standard

processing_time = 3  # ms
ack_frame_time = 0.2  # ms

frame_path_losses = extract_frames(path_loss_list, frame_interval, frame_time)
del path_loss_list
print(f"Signal loaded. {time.perf_counter() - start_time:2f}s")
print(f"Total number of frames: {total_nr_frames}")

TPC_methods: dict[str, TPCMethodInterface] = {
    "Optimal": Optimal(frame_path_losses, packet_loss_RSSI, total_nr_frames),
    "Constant": Constant(-10, total_nr_frames),
    "Xiao_aggressive": Xiao_aggressive(total_nr_frames),
    "Gao": Gao(total_nr_frames),
    "Sodhro": Sodhro(total_nr_frames),
    "Guo": Guo(total_nr_frames, frame_path_losses)
}

# Initial transmission power setup
path_loss_avg = np.average(frame_path_losses)
for method in TPC_methods.values():
    method.current_rx_power = -10 + path_loss_avg
    method.current_tx_power = -10
    method.update_internal()

# Main simulation loop
for frame_nr, frame_path_loss in enumerate(frame_path_losses):
    for name, method in TPC_methods.items():
        method.current_rx_power = method.current_tx_power + frame_path_loss

        # Check if packet is lost in the current frame
        packet_lost = method.current_rx_power < packet_loss_RSSI

        if packet_lost:
            method.consecutive_lost_frames += 1
            method.lost_frames += 1
            method.current_rx_power = -100 # Packet loss set RSSI -100 dBm
        else:
            extra_delay = method.consecutive_lost_frames * 200  # ms
            latency = frame_time + processing_time + ack_frame_time + extra_delay
            method.latencies.append(latency)
            method.consecutive_lost_frames = 0

        method.update_internal()

        match name:
            case "Optimal":
                method.update_transmission_power(P_target, -85, -80)
            case "Constant":
                method.update_transmission_power(P_target, -85, -80)
            case "Xiao_aggressive":
                method.update_transmission_power(-82.5, -85, -80)
            case "Gao":
                method.update_transmission_power(-82.5, -85, -80)
            case "Sodhro":
                method.update_transmission_power(-85, -88, -82.3)
            case "Isak":
                method.update_transmission_power(-85, -88, -82)
            case "Guo":
                method.update_transmission_power(-85, -88, -82)

        # Update method stats
        method.update_stats(frame_nr)

print(f"Main simulation loop completed. {time.perf_counter() - start_time:2f}s")

table_header = (
    f"{'Method':<16} |"
    + f"{'E_total (J)':>12} |"
    + f"{'E_avg (mW)':>12} |"
    + f"{'σ_E':>7} |"
    + f"{'P_rx_avg (dBm)':>16} |"
    + f"{'σ_P_rx':>7} |"
    + f"{'P_tx_avg (dBm)':>16} |"
    + f"{'σ_P_tx':>7} |"
    + f"{'η_loss (%)':>12} |"
    + f"{'T_avg (ms)':>12} |"
    + f"{'J (ms)':>7}"
)
print(f"Number of samples {total_nr_frames}")
print(table_header)

for name, method in TPC_methods.items():
    stats_string = calculate_stats(method, name, frame_time, total_nr_frames)
    print(stats_string)

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