import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, gamma, lognorm

# Simulation parameters
n_samples = 10000  # Number of RSSI samples
shape_weibull = 0.98  # Shape parameter from the paper
scale_weibull = 1.82  # Scale parameter from the paper
mean_path_loss = 72  # dB (as per study's median path loss suggestion)

# Generate Weibull-distributed RSSI samples
weibull_rssi = -mean_path_loss + weibull_min.rvs(shape_weibull, scale=scale_weibull, size=n_samples)

# Plot histogram of generated RSSI values
plt.figure(figsize=(8,5))
plt.hist(weibull_rssi, bins=50, density=True, alpha=0.6, color='b', label='Simulated RSSI')
plt.xlabel("RSSI (dB)")
plt.ylabel("Probability Density")
plt.title("Simulated RSSI using Weibull Distribution")
plt.legend()
plt.show()
