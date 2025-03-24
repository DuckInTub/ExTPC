from abc import ABC, abstractmethod
import numpy as np
import math

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

            min_tx_power : float
                The minimum possible transmission power dBm.

            max_tx_power : float
                The maximum possible transmission power dBm.

        Returns:
            float
                The computed next transmission power level in dBm
        """
        pass

    @abstractmethod
    def update_internal(self) -> None:
        """
        Update the internal variables and states of the TPC method.
        This can be things like updating the running average in Xiaos aggresive.
        """
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
    """
    An implementation of the energy-efficient adaptive TPC algorithm by Sodhro.
    Internal constants (averaging weights and allowed ΔP values) are defined in the class.
    The algorithm updates the running RSSI average and then computes the adjustment ΔP.
    """
    # Internal constants (set as per the paper’s recommendations)
    ALPHA1: float = 1.0  # averaging weight for good channel
    ALPHA2: float = 0.4  # averaging weight for bad channel
    POSSIBLE_DPS: list[int] = list(range(-31, 32))  # discrete ΔP values

    def __init__(self):
        super().__init__()
        # Initialize the running average (R̄) as the current rx power.
        self.R_avg: float = self.current_rx_power
        # TRH_var (the upper variable threshold) will be computed in update_internal.
        self.TRH_var: float = None  
        # We will store the target and lower threshold when next_transmitt_power is called.
        self.R_target: float = None  # RSSI target
        self.TRL: float = None       # fixed lower threshold

    def next_transmitt_power(
        self, 
        rx_target: float, 
        rx_target_low: float, 
        rx_target_high: float, 
        min_tx_power: float, 
        max_tx_power: float
    ) -> float:
        # Store the target and fixed lower threshold (assumed constant)
        self.R_target = rx_target
        self.TRL = rx_target_low
        
        # If TRH_var has not been computed yet, initialize it using rx_target_high.
        if self.TRH_var is None:
            self.TRH_var = rx_target_high

        # Determine R_latest: take current_rx_power (alternatively, self.rx_powers[-1] could be used)
        R_latest = self.current_rx_power

        # Determine R_lowest: use the minimum RSSI from the history (if available)
        if self.rx_powers:
            R_lowest = min(self.rx_powers)
        else:
            R_lowest = R_latest  # fallback if no history exists

        # Update the running average R̄ as per the paper:
        if R_latest > self.R_avg:
            self.R_avg = R_latest + (1 - self.ALPHA1) * R_lowest
        elif R_latest < self.R_avg:
            self.R_avg = R_latest + (1 - self.ALPHA2) * R_lowest
        else:
            # If equal, do nothing (per instructions)
            pass

        # Compute ΔP based on the updated R̄.
        if self.R_avg > self.TRH_var or self.R_avg < self.TRL:
            # Calculate the gap between the target and the average
            target_gap = self.R_target - self.R_avg
            # From the discrete set, filter ΔP values that exceed the target gap.
            candidates = [dp for dp in self.POSSIBLE_DPS if dp > target_gap]
            if candidates:
                # Choose the candidate that minimizes the absolute difference.
                DeltaP = min(candidates, key=lambda dp: abs(dp - target_gap))
            else:
                DeltaP = 0
        elif self.TRL <= self.R_avg <= self.TRH_var:
            DeltaP = 0
        else:
            # Should not reach here; default to no change.
            DeltaP = 0

        # Adjust the transmission power based on ΔP while respecting bounds.
        new_tx_power = self.current_tx_power + DeltaP
        if new_tx_power > max_tx_power:
            new_tx_power = max_tx_power
        elif new_tx_power < min_tx_power:
            new_tx_power = min_tx_power

        # Update the internal current transmit power.
        self.current_tx_power = new_tx_power

        return self.current_tx_power

    def update_internal(self) -> None:
        # Ensure that we have a fixed lower threshold; if not, set a default (e.g., -88 dBm).
        if self.TRL is None:
            self.TRL = -88

        if self.rx_powers:
            n = len(self.rx_powers)
            # Compute the standard deviation (s) of the RSSI samples around the running average.
            s = math.sqrt(sum((r - self.R_avg) ** 2 for r in self.rx_powers) / n)
            self.TRH_var = self.TRL + s
        # If no RSSI history exists, leave TRH_var unchanged.
