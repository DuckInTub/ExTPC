import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import norm, gamma
import scipy.io
from pathlib import Path

def packet_lost(packet_rx_power_db : float, min_rx_power_db: float =-85.0, offset_db: float=3) -> bool:
    return packet_rx_power_db <= min_rx_power_db - offset_db

def calculate_received_power(tx_power_dbm: float, path_loss_dbm : float):
    return tx_power_dbm - path_loss_dbm

def tx_power_to_mW(tx_power):
    output_power_dbm = np.array([0, -1, -3, -5, -7, -10, -15, -25])
    power_consumption_mw = np.array([31.3, 29.7, 27.4, 25.0, 22.5, 20.2, 17.9, 15.3])

    coeffs = np.polyfit(output_power_dbm, power_consumption_mw, 2)
    poly_model = np.poly1d(coeffs)
    return poly_model(tx_power)

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