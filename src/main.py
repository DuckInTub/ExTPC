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
packet_loss_RSSI = -88
offset = 3 # dB

TPC_methods : dict[str, TPCMethodInterface] = {
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

    for name, method in TPC_methods.items():
        # Calcualte the received power of the current frame.
        start_of_frame = frame_nr*frame_interval
        frame_path_losses = path_loss_list[start_of_frame:start_of_frame+math.floor(frame_time)]
        method.current_rx_power = np.average([method.current_tx_power + path_loss for path_loss in frame_path_losses])

        # Calculate weather the current frame will be lost.
        if method.current_rx_power < packet_loss_RSSI:
            method.lost_frames.append(frame_nr)

        # Add the received power for the current frame
        # to the received_power data list of each TPC method.
        method.rx_powers.append(method.current_rx_power)

        # According to the papers different methods use different ways to calculate the running average.
        # This method computes such averages and other internal state variables. 
        method.update_internal()

        # Calculate the transmisson power of the next frame
        # Add that transmission power to the tx_power data list history
        match name:
            case "Constant":
                method.current_tx_power = method.next_transmitt_power(P_target, -85, -80, -25, 0)
            case "Xiao_aggressive":
                method.current_tx_power = method.next_transmitt_power(-82.5, -85, -80, -25, 0)
            case "Gao":
                method.current_tx_power = method.next_transmitt_power(-82.5, -85, -80, -25, 0)
            case "Sodhro":
                method.current_tx_power = method.next_transmitt_power(-80, -88, -80, -25, 0)

        method.tx_powers.append(method.current_tx_power)

plt.figure(figsize=(16, 9))
# --- Section statistics ---
for name, method in TPC_methods.items():
    lost_frames = method.lost_frames
    avg_tx_power = np.average(method.tx_powers)
    pack_loss_ratio = 100*len(lost_frames) / total_nr_frames
    power_consumed = frame_time*sum(map(tx_power_to_mW, method.tx_powers))
    print(f"{name}: Packet loss {pack_loss_ratio:.2f}% with avg_tx_power: {avg_tx_power:.2f}dBm, and power consumption {power_consumed:.2f}mW")

    plt.plot(method.rx_powers, label=name)

print(f"Number of samples {len(method.rx_powers)}")
# # --- Section graphing ---
# plt.plot(path_loss_list)
plt.xlim(0, 100)
plt.ylim(-120, -60)
plt.title("Professional Matplotlib Figure", fontsize=16, fontweight="bold")
plt.xlabel("Frame nr", fontsize=14, fontweight="bold")
plt.ylabel("RSSI (dBm)", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(True)
plt.savefig("professional_figure.png", dpi=300, bbox_inches='tight')
plt.show()

