import numpy as np

def simulate_next_RSSI(new_tx_power, current_tx_power, current_rssi, path_loss_exponent=3):
    """
    Estimate new RSSI after changing transmit power in a WBAN scenario.
    
    Parameters:
    - new_tx_power (float): New transmission power in dBm.
    - current_tx_power (float): Current transmission power in dBm.
    - current_rssi (float): RSSI at the current transmission power in dBm.
    - path_loss_exponent (float): Environmental path loss exponent (default is 3 for WBAN).
    
    Returns:
    - new_rssi (float): Predicted RSSI for the new transmission power.
    """

    # Basic RSSI estimation (assuming path loss remains constant)
    estimated_rssi = current_rssi + (new_tx_power - current_tx_power)

    # Apply log-distance correction if needed
    path_loss_correction = 10 * path_loss_exponent * np.log10(abs(new_tx_power / current_tx_power))
    new_rssi = estimated_rssi - path_loss_correction

    return new_rssi

def packet_lost(packet_rx_power_db : float, min_rx_power_db=-85.0, offset_db=3):
    return packet_rx_power_db <= min_rx_power_db - offset_db

# This is litterally how it fucking works
# The matlab codes "signal" is what fraction of transmitt power is received power
def calculate_received_power(tx_power_dbm: float, path_loss_dbm : float):
    return tx_power_dbm - path_loss_dbm

def tx_power_to_mW(tx_power):
    output_power_dbm = np.array([0, -1, -3, -5, -7, -10, -15, -25])
    power_consumption_mw = np.array([31.3, 29.7, 27.4, 25.0, 22.5, 20.2, 17.9, 15.3])

    coeffs = np.polyfit(output_power_dbm, power_consumption_mw, 2)
    poly_model = np.poly1d(coeffs)
    return poly_model(tx_power)



if __name__ == "__main__":
    # Example Usage:
    current_tx = 0.01   # Current transmission power in dBm
    current_rssi = -40 # Measured RSSI at current power in dBm
    new_tx = -25        # New transmission power in dBm

    predicted_rssi = simulate_next_RSSI(new_tx, current_tx, current_rssi)
    print(f"Predicted RSSI at {new_tx} dBm: {predicted_rssi:.2f} dBm")