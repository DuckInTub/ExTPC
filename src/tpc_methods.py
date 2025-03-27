from abc import ABC, abstractmethod
import numpy as np
import math

MIN_TX_POWER = -25
MAX_TX_POWER = 0

class TPCMethodInterface(ABC):
    """
    An abstract base class defining the interface for Transmission Power Control (TPC) methods.

    This interface standardizes how different TPC algorithms should:
    - Track received and transmitted power.
    - Compute the next transmission power based on received signal strength.
    - Update internal state variables to adapt to dynamic conditions.

    Attributes:
        rx_powers : List[float]
            A history of received power values (in dBm).

        tx_powers : List[float]
            A history of transmission power values (in dBm).

        lost_frames : List[int]
            A list tracking the frames lost during transmission.

        current_rx_power : float
            The current received power in dBm.

        current_tx_power : float
            The current transmission power in dBm.
    """
    def __init__(self, nr_samples):
        """
        Initializes the TPC method with default power values and empty tracking lists.
        """
        self.rx_powers = np.zeros(nr_samples, dtype=np.int8)
        self.tx_powers = np.zeros(nr_samples, dtype=np.int8)
        self.lost_frames: int = 0
        self.latencies: list[int] = []
        self.current_rx_power: float = -60
        self.current_tx_power: float = -10
        self.consecutive_lost_frames: int = 0

    def update_stats(self, index):
        self.rx_powers[index] = self.current_rx_power
        self.tx_powers[index] = self.current_tx_power

    @abstractmethod
    def next_transmitt_power(
        self, 
        rx_target: float, 
        rx_target_low: float, 
        rx_target_high: float
    ) -> float:
        """
        Calculate and update the next transmitted power based on the internal history
        of the method and given bounds for rx_power target.

        Parameters:
            rx_target : float
                The desired received power dBm.

            rx_target_low : float
                The lowest desired received power in dBm.

            rx_target_high : float
                The highest desired received power in dBm.
        Returns:
            float
                The computed next transmission power level in dBm in range [-25, 0]
        """
        pass

    @abstractmethod
    def update_internal(self) -> None:
        """
        Update the internal variables and states of the TPC method.
        This can be things like updating the running average in Xiaos aggresive.
        """
        pass

class Stupid(TPCMethodInterface):
    def __init__(self, nr_of_samples):
        super().__init__(nr_of_samples)

    def update_internal(self):
        pass

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high):
        current_path_loss = self.current_tx_power - self.current_rx_power
        return np.clip(rx_target + current_path_loss, MIN_TX_POWER, MAX_TX_POWER)

class Constant(TPCMethodInterface):
    def __init__(self, constant_power : float, nr_of_samples):
        super().__init__(nr_of_samples)
        self.tx_power_constant = constant_power

    def update_internal(self):
        pass

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high):
        return self.tx_power_constant # dBm

class Xiao_aggressive(TPCMethodInterface):
    def __init__(self, nr_of_samples, avg_weight=0.8):
        super().__init__(nr_of_samples)
        self.avg_weight = avg_weight # α in Xiaos paper
        self.exp_avg_rx_power: float = 0.0 # R̅ in Xiaos paper

    def update_internal(self):
        self.exp_avg_rx_power = (1 - self.avg_weight)*self.exp_avg_rx_power + self.avg_weight*self.current_rx_power

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high) -> float:
        if self.exp_avg_rx_power > rx_target_high:
            delta = -1
        if self.exp_avg_rx_power < rx_target_low:
            delta = 3
        if rx_target_low <= self.exp_avg_rx_power <= rx_target_high:
            delta = 0
        return np.clip(self.current_tx_power + delta, MIN_TX_POWER, MAX_TX_POWER)

class Gao(TPCMethodInterface):
    def __init__(self, nr_of_samples, filter_coeff=0.8):
        super().__init__(nr_of_samples)
        self.filter_coeff: float = filter_coeff
        self.average_RSSI: float = 0.0
        self.tx_power_control_steps = [-3, -2, -1, 0, 1, 2, 3, 4]

    def update_internal(self):
        self.average_RSSI = self.current_rx_power + (1 - self.filter_coeff)*self.average_RSSI

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high) -> float:
        if self.average_RSSI > rx_target_high or self.average_RSSI < rx_target_low:
            args = [step for step in self.tx_power_control_steps if step > rx_target - self.average_RSSI]
            if not args:
                return self.current_tx_power
            index_of_min = np.argmin([abs(rx_target - self.average_RSSI - step) for step in args])
            delta = self.tx_power_control_steps[index_of_min]
        elif rx_target_low <= self.average_RSSI < rx_target_high:
            delta = 0
        return np.clip(self.current_tx_power + delta, MIN_TX_POWER, MAX_TX_POWER)

class Sodhro(TPCMethodInterface):

    def __init__(self, nr_samples):
        super().__init__(nr_samples)
        self.R_avg = 0
        self.R_lowest = -80 # Warning STUPID name
        self.R_latest = -80

    def update_internal(self):
        self.R_lowest = self.R_latest 
        self.R_latest = self.current_rx_power
        ALPHA_1 = 1.0
        ALPHA_2 = 0.4

        if self.R_latest > self.R_avg:
            self.R_avg = self.R_latest + (1 - ALPHA_1) * self.R_lowest
        elif self.R_latest < self.R_avg:
            self.R_avg = self.R_latest + (1 - ALPHA_2) * self.R_lowest

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high):
        P_DELTAS = list(range(-31, 32))
        R_target = rx_target
        TRH_var = rx_target_high
        TRL = rx_target_low

        if self.R_avg > TRH_var or self.R_avg < TRL:
            args = [delta for delta in P_DELTAS if delta > R_target - self.R_avg]
            if not args:
                return self.current_tx_power

            indx = np.argmin(abs(R_target - self.R_avg - arg) for arg in args)
            delta_P = args[indx]
        elif TRL <= self.R_avg <= TRH_var:
            delta_P = 0

        return np.clip(self.current_tx_power + delta_P, MIN_TX_POWER, MAX_TX_POWER)