from abc import ABC, abstractmethod
import numpy as np
import math
import collections

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

    def update_transmission_power(self, rx_target, rx_target_low, rx_target_high):
        next_P_tx = self.next_transmitt_power(rx_target, rx_target_low, rx_target_high)
        clipped = np.clip(next_P_tx, MIN_TX_POWER, MAX_TX_POWER)
        self.current_tx_power = clipped
        return clipped

    @abstractmethod
    def next_transmitt_power(
        self, 
        rx_target: float, 
        rx_target_low: float, 
        rx_target_high: float
    ) -> float:
        """
        Calculate the next transmitted power based on the internal history
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

class Isak(TPCMethodInterface):

    NR_SAMPLES = 4 # Number of samples for regression

    def __init__(self, nr_samples, packet_loss_threshhold):
        super().__init__(nr_samples)
        self.packet_loss_limit = packet_loss_threshhold
        self.latest_pathlosses = collections.deque([], maxlen=Isak.NR_SAMPLES) # FIFO length 8
        self.limit_high = 0

    def update_internal(self):
        current_path_loss = self.current_rx_power - self.current_tx_power
        self.latest_pathlosses.append(current_path_loss)
        self.limit_high = max(self.latest_pathlosses) + (1 - 0.8) * self.limit_high

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high):
        if len(self.latest_pathlosses) < Isak.NR_SAMPLES:
            return -10

        path_losses_filtered = [L if L >= -85 else -85 for L in self.latest_pathlosses]
        coeffs = np.polyfit(range(0, Isak.NR_SAMPLES, 1), path_losses_filtered, 3)
        current_model = np.poly1d(coeffs)
        predicted_path_loss = current_model(Isak.NR_SAMPLES+1)
        next_tx_power = rx_target - predicted_path_loss
        return np.clip(next_tx_power, MIN_TX_POWER, MAX_TX_POWER)


class Stupid(TPCMethodInterface):
    def __init__(self, nr_of_samples):
        super().__init__(nr_of_samples)

    def update_internal(self):
        pass

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high):
        current_path_loss = self.current_tx_power - self.current_rx_power
        return np.clip(rx_target + current_path_loss, MIN_TX_POWER, MAX_TX_POWER)


class Optimal(TPCMethodInterface):
    def calculate_optimal(frame_path_loss_list, frame_interval, frame_time, packet_loss_threshhold):
        NUMBER_FRAMES = len(frame_path_loss_list)
        tx_powers = np.zeros(NUMBER_FRAMES, dtype=np.int8)

        for frame_nr, path_loss in enumerate(frame_path_loss_list):
            if path_loss <= packet_loss_threshhold:
                tx_power = -25
            else:
                tx_power = (packet_loss_threshhold ) + abs(path_loss)
            tx_powers[frame_nr] = tx_power

        return tx_powers

    def __init__(self, path_loss_list, frame_interval, frame_time, packet_loss_RSSI, nr_of_samples):
        super().__init__(nr_of_samples)
        self.internal_optimal = Optimal.calculate_optimal(path_loss_list, frame_interval, frame_time, packet_loss_RSSI)
        self.internal_optimal = np.roll(self.internal_optimal, -1)
        self.indx = 0

    def update_internal(self):
        pass

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high):
        ret = self.internal_optimal[self.indx]
        self.indx += 1
        return np.clip(ret, MIN_TX_POWER, MAX_TX_POWER)

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

class Xiao_conservative(TPCMethodInterface):
    def __init__(self, nr_of_samples, history_N: int = 10, decrease_delta: int = -2):
        super().__init__(nr_of_samples)
        self.history_N = history_N
        self.decrease_delta = decrease_delta
        self.Th_counter: int = 0

    def update_internal(self):
        pass

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high) -> float:
        if self.current_rx_power > rx_target_high:
            self.Th_counter += 1
        else:
            self.Th_counter = 0

        if self.current_rx_power < rx_target_low:
            self.Th_counter = 0
            return MAX_TX_POWER
        delta = 0
        if self.Th_counter >= self.history_N:
            delta = self.decrease_delta
            self.Th_counter = 0
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

class Smith_2012(TPCMethodInterface):

    def __init__(self, nr_samples):
        pass
        # super().__init__(nr_samples)
        # Rx_sens = [-95, -93, -90, -86]
        # k = [1, 2, 3, 4]
        # levels_k = [-95, -92.5, -90, -46]


        # U_rms = math.sqrt( // )

        # if U_rms > -75:
        #     c = [9, 8.5, 7.5, 7]
        # elif U_rms <= -75:
        #     c = [6.5, 6.5, 5.5, 4.5]
        
        # for L in range(1, T_pr+1):
        #     i = [indx for indx in [0, 1, 2] if levels_k[indx] < S_p[L] <= levels_k[indx+1]][0]
        #     C = Rx_sens[k] + c[k]
        #     if levels_k[i] <= C:
        #         Tx_out[L] = 0
        #     elif C + 0.5 <= levels_k[i] <= C + 30:
        #         Tx_out[L] = C - levels_k[i]
        #     else:
        #         Tx_out[L] = -25

class Sodhro(TPCMethodInterface):

    def __init__(self, nr_samples):
        super().__init__(nr_samples)
        self.R_avg = 0
        self.R_lowest = -80 # Warning STUPID name. "More like next to latest"
        self.R_latest = -80
        self.N = 0 # n in eq 4 & 5 in Sodhro
        self.R_i_R_avg_sum = 0 # 
        self.TRL = -88
        self.TRH_var = 0

    def update_internal(self):
        self.R_lowest = self.R_latest 
        self.R_latest = self.current_rx_power
        ALPHA_1 = 1.0
        ALPHA_2 = 0.4

        # Update R_avg eq. (1, 2)
        if self.R_latest > self.R_avg:
            self.R_avg = self.R_latest + (1 - ALPHA_1) * self.R_lowest
        elif self.R_latest < self.R_avg:
            self.R_avg = self.R_latest + (1 - ALPHA_2) * self.R_lowest

        # Update TRH_var dynamically eq. (4, 5)
        self.R_i_R_avg_sum += self.current_rx_power - self.R_avg
        self.N += 1

        sigma = math.sqrt( (1 / self.N) * self.R_i_R_avg_sum )
        self.TRH_var = self.TRL + sigma

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high):
        P_DELTAS = list(range(-31, 32))
        R_target = rx_target

        if self.R_avg > self.TRH_var or self.R_avg < self.TRL:
            args = [delta for delta in P_DELTAS if delta > R_target - self.R_avg]
            if not args:
                return self.current_tx_power

            indx = np.argmin(abs(R_target - self.R_avg - arg) for arg in args)
            delta_P = args[indx]
        elif self.TRL <= self.R_avg <= self.TRH_var:
            delta_P = 0

        return np.clip(self.current_tx_power + delta_P, MIN_TX_POWER, MAX_TX_POWER)