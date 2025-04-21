import numpy as np
import matplotlib.pyplot as plt
import math
from tpc_methods import *
from util_functions import *
from pathlib import Path
import pickle
import time

SIMULATE = False
GRAPH_STATS = True
TIME = 300

# Parameters
frame_time = 5.2701  # ms
frame_interval = 200  # ms

tx_power_min = -25  # dBm
rx_power_max = 0  # dBm
P_target = -85  # dBm
packet_loss_RSSI = -88 # Cite the standard

frame_processing_time = 3  # ms
ack_frame_time = 0.2  # ms

start_time = time.perf_counter()
if SIMULATE:
    # Simulate data
    path_loss_list = simulate_path_loss(1000, TIME)
else:
    # Load data from pickle
    path = Path("..") / "data" / "data.pkl"
    with open(path, "rb") as file:
        data_dict = pickle.load(file)
        most_data_keys = ["pl_LeftWrist_RightHip", "pl_RightWrist_RightHip", "pl_LeftAnkle_RightHip"]
        indx = 0
        key = "pl_LeftAnkle_Chest"
        path_loss_list = []
        for person, data in data_dict[key]["9kph"].items():
            path_loss_list += data
        del data_dict


frame_path_losses = extract_frames(path_loss_list, frame_interval, frame_time)
path_loss_average = np.average(path_loss_list)
path_loss_std = np.std(path_loss_list)
total_nr_frames = len(frame_path_losses)
del path_loss_list
print(f"Signal loaded. {time.perf_counter() - start_time:2f}s")
print(f"Total number of frames: {total_nr_frames}")

INFO = (
    f"{TIME}s for {total_nr_frames} frames at frame interval {frame_interval}ms and "
    f"frame time {frame_time:.2f}ms. Average gain was {path_loss_average:.2f}dBm with "
    f"standard deviation {path_loss_std:.2f}dBm."
)
INFO = "Simulated scenario. " + INFO if SIMULATE else f"Real scenario {key}. " + INFO

TPC_methods: dict[str, TPCMethodInterface] = {
    "Optimal": Optimal(frame_path_losses, packet_loss_RSSI, total_nr_frames),
    "Constant": Constant(total_nr_frames, -10),
    "Naive": Naive(total_nr_frames),
    # "Xiao_aggr_2008": Xiao_aggressive_2008(total_nr_frames),
    # "Xiao_cons_2008": Xiao_conservative_2008(total_nr_frames),
    # "Xiao_aggr_2009": Xiao_2009(total_nr_frames, 0.2, 0.8),
    "Xiao_bal_2009": Xiao_2009(total_nr_frames, 0.8, 0.8),
    # "Xiao_cons_2009": Xiao_2009(total_nr_frames, 0.8, 0.2),
    # "Gao": Gao(total_nr_frames),
    # "Sodhro": Sodhro(total_nr_frames),
    "Guo": Guo(total_nr_frames),
    # "Smith": Smith_2011(total_nr_frames, 3),
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

print(INFO)

# --- Text stats in table ---
print(TPCMethodInterface.get_stats_header())
for name, method in TPC_methods.items():
    print(method.calculate_stats(name, frame_time, total_nr_frames))

if GRAPH_STATS:
    # --- Plot 1: Boxplot for RX powers ---
    plt.figure(1, figsize=(16, 9))
    plt.yticks(np.arange(-100, 50, 3))
    plt.title("Box plot: RX powers." + INFO, fontsize=12)
    plt.ylabel("RX Power (dBm)")
    plt.grid(True)
    data = [method.rx_powers for _, method in TPC_methods.items()]
    labels = [name for name, _ in TPC_methods.items()]
    plt.boxplot(data, tick_labels=labels)
    plt.axhline(y=P_target, color='g', linestyle='--', label="Target Average")
    plt.axhline(y=packet_loss_RSSI, color='r', linestyle='-', label="Packet loss cutoff")
    plt.legend()
    plt.show()

    # --- Plot 2: Boxplot for TX powers ---
    plt.figure(2, figsize=(16, 9))
    plt.yticks(np.arange(-25, 1, 1))
    plt.title("Box plot: TX powers." + INFO, fontsize=12)
    plt.ylabel("TX Power (dBm)")
    plt.grid(True)
    data = [method.tx_powers for _, method in TPC_methods.items()]
    labels = [name for name, _ in TPC_methods.items()]
    plt.boxplot(data, tick_labels=labels)
    plt.show()

    # --- Plot 3: Time series for RX power ---
    plt.figure(3, figsize=(16, 9))
    for name, method in TPC_methods.items():
        plt.plot(method.rx_powers, label=name)
    # Plot settings
    plt.xlim(0, total_nr_frames)
    plt.ylim(-120, -60)
    plt.title("Received power (dBm) over time." + INFO, fontsize=12)
    plt.xlabel("Frame nr", fontsize=14, fontweight="bold")
    plt.ylabel("P_rx (dBm)", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True)
    plt.show()