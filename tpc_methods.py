from abc import ABC, abstractmethod
import numpy as np
import variables
import math

class TPC_method_interface(ABC):
    def __init__(self, init_rx_power:float = variables.rx_power_target, init_tx_power: float = -25):
        # Define the parameters with their default types
        self.rx_powers: list[float] = []
        self.tx_powers: list[float] = []
        self.lost_frames: list[int] = []
        self.current_rx_power: float = init_rx_power
        self.current_tx_power: float = init_tx_power

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

    @abstractmethod
    def update_internal(self):
        """
        Update the internal variables and states of the TPC method.
        This can be things like updating the running average in Xiaos aggresive.
        """
        pass


class Constant(TPC_method_interface):
    def __init__(self, constant_power : float, init_rx_power = variables.rx_power_target, init_tx_power = -25):
        super().__init__(init_rx_power, init_tx_power)
        self.tx_power_constant = constant_power

    def update_internal(self):
        pass

    def next_transmitt_power(self, rx_power_target_low : float, rx_power_target_high : float) -> float:
        return self.tx_power_constant # dBm

class Xiao_aggressive(TPC_method_interface):
    def __init__(self, avg_weight=0.8, init_rx_power = variables.rx_power_target, init_tx_power = -25):
        super().__init__(init_rx_power, init_tx_power)
        self.avg_weight = avg_weight
        self.exp_avg_rx_power: float = 0.0 # R̅ in Xiaos paper

    def update_internal(self):
        self.exp_avg_rx_power = (1 - self.avg_weight)*self.exp_avg_rx_power + self.avg_weight*self.current_rx_power

    def next_transmitt_power(self, rx_power_target_low : float, rx_power_target_high) -> float:

        if self.exp_avg_rx_power > rx_power_target_high:
            delta = -1

        if self.exp_avg_rx_power < rx_power_target_low:
            delta = 3 # NOTE: the unit is dBm so this is a doubling (logarithmic scale)

        if rx_power_target_low < self.exp_avg_rx_power < rx_power_target_high:
            delta = 0

        return np.clip(self.current_tx_power + delta, variables.tx_power_min, variables.tx_power_max)

class Gao(TPC_method_interface):
    def __init__(self, filter_coeff=0.8, init_rx_power = variables.rx_power_target, init_tx_power = -25):
        super().__init__(init_rx_power, init_tx_power)
        self.filter_coeff: float = filter_coeff
        self.average_RSSI: float = 0.0

    def update_internal(self):
        self.average_RSSI = self.current_rx_power + (1 - self.filter_coeff)*self.average_RSSI


    def next_transmitt_power(self, rx_power_target_low : float, rx_power_target_high) -> float:
        tx_power_control_steps = [-3, -2, -1, 0, 1, 2, 3, 4]
        power_target = -82.5
        
        if rx_power_target_low < self.average_RSSI < rx_power_target_high:
            index_of_min = np.argmin([abs(power_target - self.average_RSSI - step) for step in tx_power_control_steps])
            delta = tx_power_control_steps[index_of_min]
        else:
            delta = 0

        return np.clip(self.current_tx_power + delta, variables.tx_power_min, variables.tx_power_max)

class Sodhro(TPC_method_interface):
    def __init__(self, init_rx_power = variables.rx_power_target, init_tx_power = -25):
        super().__init__(init_rx_power, init_tx_power)
        self.R_latest: float = 0.0
        self.R_lowest: float = 0.0
        self.R_avg: float = 0.0
        self.alpha_1: float = 1.0
        self.alpha_2: float = 0.4
        self.tx_power_control_steps = list(range(-31, 32))
        self.TRL = -88
        self.TRH = -83

    def update_internal(self):
        self.R_latest = self.R_lowest
        self.R_lowest = self.current_rx_power

        if self.R_latest > self.R_avg:
            self.R_avg = self.R_latest+(1-self.alpha_1)*self.R_lowest
        elif self.R_latest == self.R_avg:
            pass
        else:
            self.R_avg = self.R_latest+(1-self.alpha_2)*self.R_lowest


    def next_transmitt_power(self, rx_power_target_low: float, rx_power_target_high : float) -> float:
        if self.R_avg > self.TRH or self.R_avg < self.TRL:
            # Step 7 in sodhro Fig 8.
            index_of_min = np.argmin([math.sqrt( (variables.rx_power_target - self.R_avg - step) ** 2) for step in self.tx_power_control_steps]) 
            delta = self.tx_power_control_steps[index_of_min]
        else:
            delta = 0

        return np.clip(self.current_tx_power + delta, variables.tx_power_min, variables.tx_power_max)



