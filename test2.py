import numpy as np

def predict_rssi(new_tx_power, current_tx_power, current_rssi, path_loss_exponent=3):
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
