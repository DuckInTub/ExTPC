import numpy as np
import matplotlib.pyplot as plt
import math
import variables

from tpc_methods import *
from util_functions import *

# Load data from .mat
from pathlib import Path
path = Path("..") / "data" / "20080919-Male1_3kph.mat"
path_loss_list = load_mat_file(path)

# Parameters
frame_time = 5.2701  # ms
frame_interval = 200  # ms
total_nr_frames = math.ceil(1000 * 60 / frame_interval)

tx_power_min = -25  # dBm
rx_power_max = 0  # dBm
P_target = -85  # dBm
packet_loss_RSSI = -88
offset = 3  # dB

processing_time = 3  # ms
ack_frame_time = 0.2  # ms

TPC_methods: dict[str, TPC_method_interface] = {
    "Constant": Constant(-10),
    "Xiao_aggressive": Xiao_aggressive(),
    "Gao": Gao(),
    "Sodhro": Sodhro(),
}

# Initial transmission power setup
path_loss_avg = np.average(path_loss_list)
first_tx_power = variables.rx_power_target - path_loss_avg
for method in TPC_methods.values():
    method.current_rx_power = variables.rx_power_target
    method.current_tx_power = first_tx_power
    method.update_internal()

# Main simulation loop
for frame_nr in range(total_nr_frames):
    for name, method in TPC_methods.items():
        start_of_frame = frame_nr * frame_interval
        frame_path_losses = path_loss_list[start_of_frame:start_of_frame + math.floor(frame_time)]
        method.current_rx_power = np.average([method.current_tx_power + path_loss for path_loss in frame_path_losses])

        # Check if packet is lost in the current frame
        packet_lost = method.current_rx_power < packet_loss_RSSI
        if packet_lost:
            method.lost_frames.append(frame_nr)

        # Latency calculation
        latency = frame_time + processing_time + ack_frame_time + (200 if packet_lost else 0)
        method.latencies.append(latency)

        # Update received power
        method.rx_powers.append(method.current_rx_power)

        method.update_internal()

        # Transmission power calculation as usual
        match name:
            case "Constant":
                method.current_tx_power = method.next_transmitt_power(P_target, -85, -80, -25, 0)
            case "Xiao_aggressive":
                method.current_tx_power = method.next_transmitt_power(-82.5, -85, -80, -25, 0)
            case "Gao":
                method.current_tx_power = method.next_transmitt_power(-82.5, -85, -80, -25, 0)
            case "Sodhro":
                method.current_tx_power = method.next_transmitt_power(-85, -88, -83, -25, 0)

        method.tx_powers.append(method.current_tx_power)

# Plot and summary statistics
plt.figure(figsize=(16, 9))

for name, method in TPC_methods.items():
    avg_tx_power = np.average(method.tx_powers)
    packet_loss_ratio = 100 * len(method.lost_frames) / total_nr_frames
    power_consumed = frame_time * sum(map(tx_power_to_mW, method.tx_powers))
    avg_latency = np.average(method.latencies)

    print(f"{name}: Packet loss {packet_loss_ratio:.2f}% with avg_tx_power: {avg_tx_power:.2f} dBm, power consumption: {power_consumed:.2f} mW, avg latency: {avg_latency:.2f} ms")

    plt.plot(method.rx_powers, label=name)

print(f"Number of samples {len(method.rx_powers)}")

# Plot settings
plt.xlim(0, 100)
plt.ylim(-120, -60)
plt.title("Professional Matplotlib Figure", fontsize=16, fontweight="bold")
plt.xlabel("Frame nr", fontsize=14, fontweight="bold")
plt.ylabel("RSSI (dBm)", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(True)
plt.savefig("professional_figure.png", dpi=300, bbox_inches='tight')
plt.show()
