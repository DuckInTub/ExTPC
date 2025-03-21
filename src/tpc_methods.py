from abc import ABC, abstractmethod
import numpy as np
import math

class TPCMethodInterface(ABC):
    """
    An abstract base class defining the interface for Transmission Power Control (TPC) methods.
    ...
    """
    def __init__(self):
        """
        Initializes the TPC method with default power values and empty tracking lists.
        """
        self.rx_powers: list[float] = []
        self.tx_powers: list[float] = []
        self.lost_frames: list[int] = []
        self.latencies: list[float] = []
        self.current_rx_power: float = -60
        self.current_tx_power: float = -25
        self.consecutive_lost_frames: int = 0
        self.jitter_values: list[float] = []

    @abstractmethod
    def next_transmitt_power(
        self, 
        rx_target: float, 
        rx_target_low: float, 
        rx_target_high: float, 
        min_tx_power: float, 
        max_tx_power: float
    ) -> float:
        pass

    @abstractmethod
    def update_internal(self) -> None:
        pass

class Constant(TPCMethodInterface):
    def __init__(self, constant_power : float):
        super().__init__()
        self.tx_power_constant = constant_power

    def update_internal(self):
        pass

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high, min_tx_power, max_tx_power):
        return self.tx_power_constant # dBm

class Xiao_aggressive(TPCMethodInterface):
    def __init__(self, avg_weight=0.8):
        super().__init__()
        self.avg_weight = avg_weight # α in Xiaos paper
        self.exp_avg_rx_power: float = 0.0 # R̅ in Xiaos paper

    def update_internal(self):
        self.exp_avg_rx_power = (1 - self.avg_weight)*self.exp_avg_rx_power + self.avg_weight*self.current_rx_power

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high, min_tx_power, max_tx_power) -> float:
        if self.exp_avg_rx_power > rx_target_high:
            delta = -1
        if self.exp_avg_rx_power < rx_target_low:
            delta = 3
        if rx_target_low <= self.exp_avg_rx_power <= rx_target_high:
            delta = 0
        return np.clip(self.current_tx_power + delta, min_tx_power, max_tx_power)

class Gao(TPCMethodInterface):
    def __init__(self, filter_coeff=0.8):
        super().__init__()
        self.filter_coeff: float = filter_coeff
        self.average_RSSI: float = 0.0
        self.tx_power_control_steps = [-3, -2, -1, 0, 1, 2, 3, 4]

    def update_internal(self):
        self.average_RSSI = self.current_rx_power + (1 - self.filter_coeff)*self.average_RSSI

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high, min_tx_power, max_tx_power) -> float:
        if self.average_RSSI > rx_target_high or self.average_RSSI < rx_target_low:
            args = [step for step in self.tx_power_control_steps if step > rx_target - self.average_RSSI]
            if not args:
                return self.current_tx_power
            index_of_min = np.argmin([abs(rx_target - self.average_RSSI - step) for step in args])
            delta = self.tx_power_control_steps[index_of_min]
        elif rx_target_low <= self.average_RSSI < rx_target_high:
            delta = 0
        return np.clip(self.current_tx_power + delta, min_tx_power, max_tx_power)

class Sodhro(TPCMethodInterface):
    def __init__(self):
        super().__init__()
        self.R_latest: float = self.current_rx_power
        self.R_lowest: float = self.current_rx_power
        self.R_avg: float = 0.0
        self.alpha_1: float = 1.0
        self.alpha_2: float = 0.4
        self.tx_power_control_steps = list(range(-31, 32))

    def update_internal(self):
        self.R_latest = self.R_lowest
        self.R_lowest = self.current_rx_power
        if self.R_latest > self.R_avg:
            self.R_avg = self.R_latest + (1 - self.alpha_1)*self.R_lowest
        if self.R_latest < self.R_avg:
            self.R_avg = self.R_latest + (1 - self.alpha_2)*self.R_lowest

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high, min_tx_power, max_tx_power) -> float:
        TRH = rx_target_high
        TRL = rx_target_low
        R_target = rx_target
        if self.R_avg > TRH or self.R_avg < TRL:
            args = [step for step in self.tx_power_control_steps if step > R_target - self.R_avg]
            if not args:
                return self.current_tx_power
            index_of_min = np.argmin([math.sqrt((R_target - self.R_avg - step)**2) for step in args])
            delta = self.tx_power_control_steps[index_of_min]
        else:
            delta = 0
        return np.clip(self.current_tx_power + delta, min_tx_power, max_tx_power)
