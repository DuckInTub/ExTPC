import numpy as np
import matplotlib.pyplot as plt
import math
from tpc_methods import *
from util_functions import *
from pathlib import Path

# Load data from .mat
path = Path("..") / "data" / "20080919-Male1_3kph.mat"
path_loss_list = load_mat_file(path)
# path_loss_list = simulate_path_loss(1000, 60)

# Parameters
frame_time = 5.2701  # ms
frame_interval = 200  # ms
total_nr_frames = len(path_loss_list) // frame_interval
print(f"Total number of frames: {total_nr_frames}")

tx_power_min = -25  # dBm
rx_power_max = 0  # dBm
P_target = -85  # dBm
packet_loss_RSSI = -88
offset = 3  # dB

processing_time = 3  # ms
ack_frame_time = 0.2  # ms

TPC_methods: dict[str, TPCMethodInterface] = {
    "Optimal": Optimal(path_loss_list, frame_interval, frame_time, packet_loss_RSSI, total_nr_frames),
    "Constant": Constant(-10, total_nr_frames),
    "Xiao_aggressive": Xiao_aggressive(total_nr_frames),
    "Gao": Gao(total_nr_frames),
    "Sodhro": Sodhro(total_nr_frames),
    "Isak" : Isak(total_nr_frames)
}

# Initial transmission power setup
path_loss_avg = np.average(path_loss_list)
for method in TPC_methods.values():
    method.current_rx_power = -10 + path_loss_avg
    method.current_tx_power = -10
    method.update_internal()

# Main simulation loop
for frame_nr in range(total_nr_frames):
    for name, method in TPC_methods.items():
        start_of_frame = frame_nr * frame_interval
        frame_path_losses = path_loss_list[start_of_frame:start_of_frame + math.floor(frame_time)]
        path_loss = np.average(frame_path_losses)
        opt = packet_loss_RSSI - path_loss + 0
        assert opt + path_loss >= packet_loss_RSSI
        method.current_rx_power = method.current_tx_power + path_loss

        # Check if packet is lost in the current frame
        packet_lost = method.current_rx_power < packet_loss_RSSI

        if packet_lost:
            method.consecutive_lost_frames += 1
            method.lost_frames += 1
        else:
            extra_delay = method.consecutive_lost_frames * 200  # ms
            latency = frame_time + processing_time + ack_frame_time + extra_delay
            method.latencies.append(latency)
            method.consecutive_lost_frames = 0

        method.update_internal()

        # Transmission power calculation as usual
        match name:
            case "Optimal":
                method.current_tx_power = method.next_transmitt_power(P_target, -85, -80)
            case "Constant":
                method.current_tx_power = method.next_transmitt_power(P_target, -85, -80)
            case "Xiao_aggressive":
                method.current_tx_power = method.next_transmitt_power(-82.5, -85, -80)
            case "Gao":
                method.current_tx_power = method.next_transmitt_power(-82.5, -85, -80)
            case "Sodhro":
                method.current_tx_power = method.next_transmitt_power(-85, -88, -82.3) # Values come from precalculation
            case "Isak":
                method.current_tx_power = method.next_transmitt_power(-85, -88, -82)

        # Update method stats
        method.update_stats(frame_nr)


table_header = (
    f"{'Method':<20} |"
    + f"{'E_total (J)':>12} |"
    + f"{'E_avg (mW)':>12} |"
    + f"{'σ_E':>12} |"
    + f"{'P_rx_avg (dBm)':>16} |"
    + f"{'σ_P_rx':>12} |"
    + f"{'P_tx_avg (dBm)':>16} |"
    + f"{'σ_P_tx':>12} |"
    + f"{'η_loss (%)':>12} |"
    + f"{'T_avg (ms)':>12} |"
    + f"{'J (ms)':>12}"
)
print(table_header)

# Plot and summary statistics
plt.figure(figsize=(16, 9))
for name, method in TPC_methods.items():
    stats_string = calculate_stats(method, name, frame_time, total_nr_frames)
    print(stats_string)
    plt.plot(method.rx_powers, label=name)

print(f"Number of samples {total_nr_frames}")
# Plot settings
plt.xlim(0, total_nr_frames)
plt.ylim(-120, -60)
plt.title("Professional Matplotlib Figure", fontsize=16, fontweight="bold")
plt.xlabel("Frame nr", fontsize=14, fontweight="bold")
plt.ylabel("RSSI (dBm)", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(True)