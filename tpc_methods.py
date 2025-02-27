from abc import ABC, abstractmethod
import numpy as np
import variables

class TPC_method_interface(ABC):
    def __init__(self):
        # Define the parameters with their default types
        self.rx_powers: list[float] = []
        self.tx_powers: list[float] = []
        self.lost_frames: list[int] = []
        self.current_rx_power: float = 0.0
        self.current_tx_power: float = 0.0
        self.average_received_power: float = 0.0
        self.exp_avg_rx_power: float = 0.0 # R̅ in Xiaos paper


    @abstractmethod
    def next_transmitt_power(self, rx_power_target_low : float, rx_power_target_high : float) -> float:
        """
        Calculate and update the next transmitted power based on the internal history
        of the method and given bounds for rx_power target.

        Parameters:
        ----------
        rx_power_target_low : float
            The lower bound of the desired received power range dBm.
    
        rx_power_target_high : float
            The upper bound of the desired received power range dBm.

        Returns:
        -------
        float
            The computed next transmission power level in dBm
        """
        pass

class Constant(TPC_method_interface):
    def __init__(self, constant_power_dB):
        super().__init__()
        self.tx_power_constant = constant_power_dB


    def next_transmitt_power(self, current_received_power: float) -> float:
        return self.tx_power_constant # dBm

class Xiao_aggressive(TPC_method_interface):
    def __init__(self, constant_power_dB):
        super().__init__()
        self.tx_power_constant = constant_power_dB


    def next_transmitt_power(self, rx_power_target_low : float, rx_power_target_high) -> float:

        if self.exp_avg_rx_power > rx_power_target_high:
            delta = -1

        if self.exp_avg_rx_power < rx_power_target_low:
            delta = 3 # NOTE: the unit is dBm so this is a doubling (logarithmic scale)

        if rx_power_target_low < self.exp_avg_rx_power < rx_power_target_high:
            delta = 0

        return np.clip(self.current_tx_power + delta, variables.tx_power_min, variables.tx_power_max)

class Gao(TPC_method_interface):
    def __init__(self, constant_power_dB):
        super().__init__()
        self.tx_power_constant = constant_power_dB


    def next_transmitt_power(self, rx_power_target_low : float, rx_power_target_high) -> float:
        tx_power_control_steps = [-3, -2, -1, 0, 1, 2, 3, 4]
        
        if rx_power_target_low < self.exp_avg_rx_power < rx_power_target_high:
            index_of_min = np.argmin([abs(variables.rx_power_target - self.exp_avg_rx_power - step) for step in tx_power_control_steps])
            delta = tx_power_control_steps[index_of_min]

        return np.clip(self.current_tx_power + delta, variables.tx_power_min, variables.tx_power_max)

class Sodhro(TPC_method_interface):
    def __init__(self, constant_power_dB):
        super().__init__()
        self.tx_power_constant = constant_power_dB


    def next_transmitt_power(self, current_received_power: float) -> float:
        return self.tx_power_constant # dBm
