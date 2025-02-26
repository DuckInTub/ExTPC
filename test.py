import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, gamma

# Simulation parameters
sampling_rate = 1000  # 1 kHz
duration = 60  # 60 seconds
num_samples = sampling_rate * duration  # Total number of RSSI values

# Path loss parameters (from the paper)
mean_path_loss_dB = -72  # dBm
doppler_spread = 3.5  # Hz

# Weibull distribution for fade durations
weibull_shape = 0.978  # Shape parameter for walking at 2.4 GHz
weibull_scale = 1.82  # Scale parameter for walking at 2.4 GHz

# Generate fade durations in samples (convert time to index units)
fade_durations_time = weibull_min.rvs(weibull_shape, scale=weibull_scale, size=num_samples // 500)
fade_durations_samples = np.clip((fade_durations_time * sampling_rate).astype(int), 1, 3000)  # 1 ms to 3 sec

# Gamma distribution for fade magnitudes
gamma_shape = 1.31  # From Table 8 (2.4 GHz)
gamma_scale = 0.562  # From Table 8 (2.4 GHz)

# Generate fade magnitudes in dB
fade_magnitudes = gamma.rvs(gamma_shape, scale=gamma_scale, size=len(fade_durations_samples))

# Initialize RSSI series
rssi_values = np.full(num_samples, mean_path_loss_dB)

# Apply fades according to fade durations
current_index = 0
for i in range(len(fade_durations_samples)):
    fade_duration = fade_durations_samples[i]
    fade_magnitude = fade_magnitudes[i]

    if current_index + fade_duration >= num_samples:
        break  # Prevent overflow

    # Apply fade over duration
    rssi_values[current_index:current_index + fade_duration] -= fade_magnitude
    current_index += fade_duration  # Move to next fade event

# Generate Doppler effect (small-scale fading)
time = np.linspace(0, duration, num_samples)
doppler_fading = np.cos(2 * np.pi * doppler_spread * time + np.random.uniform(0, 2*np.pi))

# Apply Doppler effect
rssi_values += doppler_fading

# Plot the generated RSSI values
plt.figure(figsize=(10, 4))
plt.plot(time[:5000], rssi_values[:5000], label="Simulated RSSI (First 5s)")
plt.xlabel("Time (s)")
plt.ylabel("RSSI (dBm)")
plt.title("Simulated RSSI for Walking Scenario in WBAN")
plt.legend()
plt.grid()
plt.show()

# Store RSSI in an array
rssi_array = rssi_values
rssi_array[:10]  # Preview first 10 values
