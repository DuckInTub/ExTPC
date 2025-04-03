import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import norm, gamma
import scipy.io
from pathlib import Path
import math
from tpc_methods import TPCMethodInterface

def calculate_received_power(tx_power_dbm: float, path_loss_dbm : float):
    return tx_power_dbm - path_loss_dbm

def tx_power_to_mW(tx_power):
    output_power_dbm = np.array([0, -1, -3, -5, -7, -10, -15, -25])
    power_consumption_mw = np.array([31.3, 29.7, 27.4, 25.0, 22.5, 20.2, 17.9, 15.3])

    coeffs = np.polyfit(output_power_dbm, power_consumption_mw, 2)
    poly_model = np.poly1d(coeffs)
    return poly_model(tx_power)


def calculate_stats(method : TPCMethodInterface, name : str, frame_time, total_nr_frames):
    map_thing = list(map(tx_power_to_mW, method.tx_powers))
    total_consumed_power = frame_time * sum(map_thing)
    total_consumed_power /= 1000
    avg_tx_power = np.average(method.tx_powers)
    std_tx_power = np.std(method.tx_powers)
    avg_consumed_power = tx_power_to_mW(avg_tx_power)
    std_power_consumption = np.std(map_thing)
    avg_rx_power = np.average(method.rx_powers)
    std_rx_power = np.std(method.rx_powers)
    packet_loss_ratio = 100 * method.lost_frames / total_nr_frames
    
    avg_latency, jitter = "N/A", "N/A"
    if method.latencies:
        avg_latency = np.average(method.latencies)
        jitter = np.std(method.latencies)
    
    return (
        f"{name:<20} |"
        + f"{total_consumed_power:>12.3f} |"
        + f"{avg_consumed_power:>12.3f} |"
        + f"{std_power_consumption:>12.3f} |"
        + f"{avg_rx_power:>16.2f} |"
        + f"{std_rx_power:>12.2f} |"
        + f"{avg_tx_power:>16.2f} |"
        + f"{std_tx_power:>12.2f} |"
        + f"{packet_loss_ratio:>12.2f} |"
        + f"{avg_latency:>12.2f} |"
        + f"{jitter:>12.2f}"
    )

def simulate_path_loss(sample_rate : int, time : float) -> np.ndarray:
    """
    Simulate a smooth path loss time series for a WBAN on-body walking scenario at 820 MHz.
    
    The function returns an array of length (sample_rate * time) where each entry is a 
    path loss in dB. The simulation is based on:
    
    1. Sampling white Gaussian noise from the standard normal distribution.
    2. Passing the noise through a Butterworth low-pass filter with a cutoff of 4Hz according to \cite{smith_first-_2011}.
    3. Normalizing the filtered signal to zero mean and unit variance.
    4. Converting the Gaussian process to a uniform process using the standard normal cumulative distribution function. 
    5. Mapping the uniform process onto the desired distribution from \cite{smith_first-_2011} using the distribution's percent point function. 
    6. Convert the gamma distributed samples to dB and center it around a desired mean path loss.
    
    Parameters:
      sample_rate (float): Number of samples per second.
      time (float): Duration in seconds.
    
    Returns:
      np.ndarray: Array of simulated path loss values in dB (each a negative dB value).
    """
    N = int(sample_rate * time) # Number of samples
    
    gen = np.random.default_rng()
    wn = gen.standard_normal(N + 2*sample_rate)
    
    # Design a 4th-order Butterworth low-pass filter with a cutoff of 4 Hz
    fc = 4.0  # cutoff frequency in Hz
    b, a = butter(4, fc / (sample_rate / 2))
    
    # Apply zero-phase filtering to obtain a smooth (correlated) Gaussian process
    smooth_gauss = filtfilt(b, a, wn)[2*sample_rate:]
    
    # Normalize to zero mean and unit variance
    smooth_gauss = (smooth_gauss - np.mean(smooth_gauss)) / np.std(smooth_gauss)
    
    # Convert the Gaussian process to a uniform process via the standard normal CDF
    uniform_vals = norm.cdf(smooth_gauss)
    
    # Map the uniform values to Gamma-distributed samples
    # Parameters based on the paper: shape a=3.52, scale b=0.251
    shape = 3.52
    scale = 0.251
    gamma_samples = gamma.ppf(uniform_vals, a=shape, scale=scale)
    
    # Convert the Gamma-distributed amplitude to dB (20*log10 for amplitude)
    fading_dB = 20 * np.log10(gamma_samples)
    
    # Add a baseline path loss offset (choose based on your reference; here we use -60 dB)
    baseline_loss = -60
    path_loss = baseline_loss + fading_dB
    
    return path_loss

def load_mat_file(mat_file_path : Path):
    import scipy.io
    """
    Load a MATLAB .mat file, list available fields, promt for selection of filed.

    Parameters:
    ----------
    mat_file : Path
        Path to the .mat file.

    Returns:
    -------
    np.ndarray
        The selected field's data as a NumPy array.
    """
    mat = scipy.io.loadmat(str(mat_file_path))
    
    data = mat["data"]
    
    field_names = list(data.dtype.names)
    
    print("Available Fields:")
    for idx, field in enumerate(field_names, 1):
        print(f"  {idx}. {field}")

    while True:
        try:
            choice = int(input("Select a field by entering its number: ")) - 1
            if 0 <= choice < len(field_names):
                selected_field = field_names[choice]
                print(f"\nYou selected: {selected_field}")
                return data[selected_field][0, 0].flatten()  # Extract and flatten the data
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

def navigate_dict(data_dict):
    """
    Walks through a nested dictionary interactively and returns the selected data array.
    :param data_dict: The dictionary to navigate.
    :return: The selected data array.
    """
    current_level = data_dict
    path = []
    
    while isinstance(current_level, dict):
        print("\nAvailable keys:")
        for i, key in enumerate(current_level.keys()):
            print(f"{i+1}: {key}")
        
        try:
            choice = int(input("Enter the number of the key to navigate: "))
            selected_key = list(current_level.keys())[choice-1]
        except (ValueError, IndexError):
            print("Invalid choice. Please enter a valid number.")
            continue
        
        path.append(selected_key)
        current_level = current_level[selected_key]
    
    print("\nFinal selection:")
    print(" -> ".join(path))
    return current_level


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    sample_rate = 1000  # e.g., 1000 samples per second
    duration = 60       # simulate for 10 seconds
    pl = simulate_path_loss(sample_rate, duration)

    from util_functions import *
    from pathlib import Path
    path_loss_list = load_mat_file(Path("..") / "data/20080919-Male1_3kph.mat")

    plt.figure(figsize=(10, 4))  # Ensure consistent figure size
    plt.plot(path_loss_list, color='b', linewidth=1, label="Real")
    # plt.plot(pl, color='#AAAA00', linewidth=1, label="Simulated")
    plt.xlim(0, 10_000),
    plt.ylim(-100, -50)
    plt.xlabel("Time (ms)", fontsize=14)
    plt.ylabel("Gain (dBm)", fontsize=14) 
    plt.grid(True)
    plt.title("Real", fontsize=16, fontweight="bold")
    plt.legend()
    plt.savefig("professional_figure.png", dpi=300, bbox_inches='tight')