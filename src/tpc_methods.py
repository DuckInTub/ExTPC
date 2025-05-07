from abc import ABC, abstractmethod
import numpy as np
import math
import collections
from util_functions import tx_power_to_mW

def level_to_tx_power(lvl : int):
    coeffs = [-0.000133168, 0.0110283, -0.338512, 5.07056, -37]
    poly_model = np.poly1d(coeffs)
    return poly_model(lvl)

MIN_TX_POWER = -25
MAX_TX_POWER = 0

class TPCMethodInterface(ABC):
    """
    An abstract base class defining the interface for 
    Transmission Power Control (TPC) methods.

    This class supervises:
    - Handling historical RX/TX data.
    - Containing the TX power to within [MIN_TX_POWER, MAX_TX_POWER].
    - Tracking current RX and TX powers.

    Subclasses must implement the `next_transmitt_power` method.

    Attributes:
        _rx_powers (np.ndarray): Stores received power values.
        _tx_powers (np.ndarray): Stores corresponding TX for each value in _rx_powers.
        _lost_frames (int): Total number of lost frames.
        _latencies (list[int]): List of calculated latencies (in ms).
        _consecutive_lost_frames (int): Number of consecutive lost frames.
        _current_rx_power (float): Most recent received power in dBm.
        _current_tx_power (float): Most recent transmitted power in dBm. 
    """
    def __init__(
            self, nr_samples, initial_tx_power: float = -10.0,
            initial_rx_power: float = -80.0
    ):
        """
        Initializes the TPC supervisor state.

        Args:
            nr_samples (int): Number of samples to store for RX and TX power.
            initial_tx_power (float): Initial transmission power in dBm.
            initial_rx_power (float): Initial received power in dBm.
        """
        if not isinstance(nr_samples, int) or nr_samples <= 0:
            raise ValueError("nr_samples must be a positive integer.")
        if not (MIN_TX_POWER <= initial_tx_power <= MAX_TX_POWER):
            raise ValueError(
                f"initial_tx_power ({initial_tx_power}) must be between "
                f"{MIN_TX_POWER} and {MAX_TX_POWER} dBm."
            )

        self._rx_powers_would = np.zeros(nr_samples, dtype=np.float32)
        self._rx_powers = np.zeros(nr_samples, dtype=np.float32)
        self._tx_powers = np.zeros(nr_samples, dtype=np.float32)
        self._lost_frames: int = 0
        self._latencies: list[int] = []
        self._consecutive_lost_frames: int = 0

        self._current_rx_power: float = initial_rx_power
        self._current_tx_power: float = initial_tx_power

    @property
    def current_rx_power(self) -> float:
        """Returns the current received power in dBm."""
        return self._current_rx_power

    @property
    def current_tx_power(self) -> float:
        """Returns the current transmission power in dBm."""
        return self._current_tx_power

    @property
    def rx_powers(self) -> float:
        """Returns the historical array of received powers in dBm."""
        return self._rx_powers

    @property
    def tx_powers(self) -> float:
        """Returns the historical array of transmitted powers in dBm."""
        return self._tx_powers

    @property
    def rx_powers_would(self) -> float:
        """Returns the array of rx powers in dBm without considering packet loss."""
        return self._rx_powers_would

    def update_current_rx(self, rx_power: float) -> None:
        """
        Updates the current received power.

        Args:
            rx_power (float): Received power in dBm.
        """
        self._current_rx_power = rx_power

    def track_stats(
            self,
            index: int,
            lost: bool,
            frame_interval_ms: int,
            frame_processing_time_ms: int,
            ack_frame_time_ms: int
            ) -> None:
        """
        Records the outcome of a transmission attempt and updates statistics.

        Args:
            index (int): Time step index.
            lost (bool): Whether the frame was lost.
            frame_interval_ms (int): Interval between frames in ms.
            frame_processing_time_ms (int): Time to process a frame in ms.
            ack_frame_time_ms (int): Time for ACK frame in ms.
        """
        would_have_been_rx = self._current_rx_power
        if lost:
            self._consecutive_lost_frames += 1
            self._lost_frames += 1
            self.update_current_rx(-100)
        else:
            extra_delay = self._consecutive_lost_frames * frame_interval_ms
            latency = (
                frame_processing_time_ms
                + frame_processing_time_ms
                + ack_frame_time_ms
                + extra_delay
            )
            self._latencies.append(latency)
            self._consecutive_lost_frames = 0

        self._rx_powers_would[index] = would_have_been_rx
        self._rx_powers[index] = self._current_rx_power
        self._tx_powers[index] = self._current_tx_power

    @classmethod
    def get_stats_header(cls) -> str:
        """
        Returns the formatted header for the stats table.

        Returns:
            str: Table header string.
        """
        table_header = (
            f"{'Method':<16} &"
            + f"{'E_total (J)':>12} &"
            + f"{'E_avg (mW)':>11} &"
            + f"{'σ_E':>6} &"
            + f"{'P_rx_avg (dBm)':>15} &"
            + f"{'σ_P_rx':>7} &"
            + f"{'P_tx_avg (dBm)':>15} &"
            + f"{'σ_P_tx':>7} &"
            + f"{'η_loss (%)':>11} &"
            + f"{'T_avg (ms)':>11} &"
            + f"{'J (ms)':>7}"
        )
        return table_header

    def calculate_stats(self, name : str, frame_time_ms : float, total_nr_frames) -> str:
        """
        Calculates and returns a formatted string of statistics.

        Args:
            name (str): Name of the TPC method.
            frame_time_ms (int): Time per frame in milliseconds.
            total_nr_frames (int): Total number of frames transmitted.

        Returns:
            str: Formatted stats row.
        """
        total_consumed_power = frame_time_ms * np.sum(tx_power_to_mW(self._tx_powers))
        total_consumed_power /= 1000
        avg_tx_power = np.average(self._tx_powers)
        std_tx_power = np.std(self._tx_powers, dtype=np.float64)
        mW_powers = tx_power_to_mW(self.tx_powers)
        avg_consumed_power = np.average(mW_powers)
        std_power_consumption = np.std(mW_powers)
        avg_rx_power = np.average(self._rx_powers)
        std_rx_power = np.std(self._rx_powers, dtype=np.float64)
        packet_loss_ratio = 100 * self._lost_frames / total_nr_frames

        avg_latency, jitter = "N/A", "N/A"
        if self._latencies:
            avg_latency = np.average(self._latencies)
            jitter = np.std(self._latencies)

        return (
            f"{name:<16} &"
            + f"{total_consumed_power:>12.3f} &"
            + f"{avg_consumed_power:>11.3f} &"
            + f"{std_power_consumption:>6.3f} &"
            + f"{avg_rx_power:>15.2f} &"
            + f"{std_rx_power:>7.2f} &"
            + f"{avg_tx_power:>15.2f} &"
            + f"{std_tx_power:>7.2f} &"
            + f"{packet_loss_ratio:>11.2f} &"
            + f"{avg_latency:>11.2f} &"
            + f"{jitter:>7.2f}"
        )

    def output_stats(self, name : str, frame_time_ms : int, total_nr_frames) -> dict:
        """Output the TPC algorithms stats as a dictionary item intended to be stored
        on disk for use in creating graphs"""
        plr = 100 * self._lost_frames / total_nr_frames
        mW_powers = tx_power_to_mW(self.tx_powers)
        avg_consumed_power = np.average(mW_powers)
        std_power_consumption = np.std(mW_powers)
        ret = {
            "name": name,
            "rx": self.rx_powers,
            "tx": self.tx_powers,
            "rx_would": self.rx_powers_would,
            "PLR": plr,
            "jitter": np.std(self._latencies),
            "avg_latency": np.average(self._latencies),
            "avg_E": avg_consumed_power,
            "std_E": std_power_consumption
        }
        return ret

    def update_transmission_power(self, rx_target, rx_target_low, rx_target_high):
        """
        Computes and updates the next transmission power.

        Args:
            rx_target (float): Target received power in dBm.
            rx_target_low (float): Lower bound of acceptable received power in dBm.
            rx_target_high (float): Upper bound of acceptable received power in dBm.

        Returns:
            float: New clipped transmission power in dBm.
        """
        next_P_tx = self.next_transmitt_power(rx_target, rx_target_low, rx_target_high)
        clipped = np.clip(next_P_tx, MIN_TX_POWER, MAX_TX_POWER)
        self._current_tx_power = clipped
        return clipped

    @abstractmethod
    def next_transmitt_power(
        self, 
        rx_target: float, 
        rx_target_low: float, 
        rx_target_high: float
    ) -> float:
        """
        Computes the next transmission power based on the algorithm.

        Args:
            rx_target (float): Target received power in dBm.
            rx_target_low (float): Lower bound of target range in dBm.
            rx_target_high (float): Upper bound of target range in dBm.

        Returns:
            float: Next transmission power in dBm.
        """
        pass

class Guo(TPCMethodInterface):

    def __init__(self, nr_samples):
        super().__init__(nr_samples)
        self.alpha = 0.2
        self.beta = 0.3
        self.num_states = 7

        self.previous_predicted_state = 2
        self.previous_state = 2

        # Counts states transitions from row i+1 to col j+1
        self.transition_counts = np.zeros((self.num_states, self.num_states))
        # Counts occurence of state i+1
        self.state_counts = np.zeros(self.num_states)
        self.prediction_matrix = np.zeros((self.num_states, self.num_states))
        self.prediction_matrix.fill(1 / self.num_states)

    @staticmethod
    def gain_to_state(gain: float) -> int:
        """ Maps channel gain (dBm) to a state (1-7) -> C(s) """
        if gain <= -88: return 1
        elif -88 < gain <= -83: return 2
        elif -83 < gain <= -78: return 3
        elif -78 < gain <= -76: return 4
        elif -76 < gain <= -73: return 5
        elif -73 < gain <= -68: return 6
        else: return 7

    @staticmethod
    def state_to_tx(state: int) -> int:
        """ Maps a state (1-7) to Tx power (dBm) -> constraint(Floor(Ĉ(s+1))) """
        return [0, 0, -5, -10, -12, -15, -20][state-1]

    def markov_predict(self, state):
        transitions_probabilities = self.prediction_matrix[state-1, :]
        possible_states = np.arange(1, self.num_states+1)
        markov_predicted_state = np.dot(transitions_probabilities, possible_states)
        return markov_predicted_state

    def update_prediction_matrix(self):
        current_gain = self.current_rx_power - self.current_tx_power
        current_state = self.gain_to_state(current_gain)

        prev_idx = self.previous_state - 1
        curr_idx = current_state - 1

        self.state_counts[curr_idx] += 1
        self.transition_counts[prev_idx, curr_idx] += 1

        M = self.state_counts[curr_idx]

        L_row = self.transition_counts[curr_idx, :]
        self.prediction_matrix[curr_idx, :] = L_row / M

        self.previous_state = current_state

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high):
        self.update_prediction_matrix()
        # C : current_state
        # C_bar : previous_predicted_state
        # C_hat : current_markov_prediction

        current_gain = self.current_rx_power - self.current_tx_power

        current_state = self.gain_to_state(current_gain)

        current_markov_prediction = self.markov_predict(current_state)

        if current_markov_prediction < 1:
            predicted_next_state = (self.previous_predicted_state + current_state) / 2
        else:
            predicted_next_state = (
                self.alpha * self.previous_predicted_state 
                + self.beta * current_state 
                + (1 - self.alpha - self.beta) * current_markov_prediction
            )

        predicted_next_state = math.floor(predicted_next_state)
        self.previous_predicted_state = predicted_next_state

        next_tx_power = self.state_to_tx(predicted_next_state)
        return next_tx_power

class Naive(TPCMethodInterface):
    def __init__(self, nr_of_samples):
        super().__init__(nr_of_samples)

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high):
        current_path_loss = self.current_tx_power - self.current_rx_power
        prev_optimal_tx =  rx_target + current_path_loss
        return prev_optimal_tx

class Optimal(TPCMethodInterface):
    def calculate_optimal(frame_path_loss_list, packet_loss_threshhold):
        NUMBER_FRAMES = len(frame_path_loss_list)
        tx_powers = np.zeros(NUMBER_FRAMES, dtype=np.int8)

        for frame_nr, path_loss in enumerate(frame_path_loss_list):
            if path_loss <= packet_loss_threshhold:
                tx_power = -25
            else:
                tx_power = (packet_loss_threshhold ) + abs(path_loss)
            tx_powers[frame_nr] = tx_power

        return tx_powers

    def __init__(self, frame_path_loss_list, packet_loss_RSSI, nr_of_samples):
        super().__init__(nr_of_samples)
        self.internal_optimal = Optimal.calculate_optimal(frame_path_loss_list, packet_loss_RSSI)
        self.internal_optimal = np.roll(self.internal_optimal, -1)
        self.indx = 0

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high):
        ret = self.internal_optimal[self.indx]
        self.indx += 1
        return np.clip(ret, MIN_TX_POWER, MAX_TX_POWER)

class Constant(TPCMethodInterface):
    def __init__(self, nr_of_samples, constant_power : float):
        super().__init__(nr_of_samples)
        self.tx_power_constant = constant_power

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high):
        return self.tx_power_constant # dBm

class Xiao_aggressive_2008(TPCMethodInterface):
    def __init__(self, nr_of_samples, avg_weight=0.8):
        super().__init__(nr_of_samples)
        self.avg_weight = avg_weight # α in Xiaos paper
        self.exp_avg_rx_power: float = 0.0 # R̅ in Xiaos paper

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high) -> float:
        self.exp_avg_rx_power = (1 - self.avg_weight)*self.exp_avg_rx_power + self.avg_weight*self.current_rx_power

        if self.exp_avg_rx_power > rx_target_high:
            delta = -1
        if self.exp_avg_rx_power < rx_target_low:
            delta = 3
        if rx_target_low <= self.exp_avg_rx_power <= rx_target_high:
            delta = 0
        return np.clip(self.current_tx_power + delta, MIN_TX_POWER, MAX_TX_POWER)

class Xiao_conservative_2008(TPCMethodInterface):
    def __init__(self, nr_of_samples, history_N: int = 10, decrease_delta: int = -2):
        super().__init__(nr_of_samples)
        self.history_N = history_N
        self.decrease_delta = decrease_delta
        self.Th_counter: int = 0

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
    
class Xiao_2009(TPCMethodInterface):
    def __init__(self, nr_samples, alpha_d, alpha_u):
        super().__init__(nr_samples)
        self.R_avg = -85
        self.alpha_d = alpha_d
        self.alpha_u = alpha_u

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high):
        R = self.current_rx_power
        if R <= self.R_avg:
            self.R_avg = self.alpha_d * R + (1 - self.alpha_d) * self.R_avg
        else:
            self.R_avg = self.alpha_u * R + (1 - self.alpha_u) * self.R_avg

        delta = 0
        if self.R_avg < rx_target_low:
            delta = 3
        elif self.R_avg > rx_target_high:
            delta = -1
        return np.clip(self.current_tx_power + delta, MIN_TX_POWER, MAX_TX_POWER)

class Xiao_2009_level(TPCMethodInterface):
    """Xiao_2009 that tries to adhere to transmitt power levels of CC2420"""
    def __init__(self, nr_samples, alpha_d, alpha_u):
        super().__init__(nr_samples)
        self.R_avg = -85
        self.alpha_d = alpha_d
        self.alpha_u = alpha_u
        self.TX_level = 31

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high):
        R = self.current_rx_power
        if R <= self.R_avg:
            self.R_avg = self.alpha_d * R + (1 - self.alpha_d) * self.R_avg
        else:
            self.R_avg = self.alpha_u * R + (1 - self.alpha_u) * self.R_avg

        if self.R_avg < rx_target_low:
            self.TX_level *= 2
        elif self.R_avg > rx_target_high:
            self.TX_level -= 2
        self.TX_level = np.clip(self.TX_level, 3, 31)
        return np.clip(level_to_tx_power(self.TX_level), MIN_TX_POWER, MAX_TX_POWER)

class Gao(TPCMethodInterface):
    def __init__(self, nr_of_samples, filter_coeff=0.8):
        super().__init__(nr_of_samples)
        self.filter_coeff: float = filter_coeff
        self.average_RSSI: float = 0.0
        self.DELTA_P_i = np.array([-3, -2, -1, 0, 1, 2, 3, 4])

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high) -> float:
        self.average_RSSI = self.filter_coeff*self.current_rx_power + (1 - self.filter_coeff)*self.average_RSSI
        # self.average_RSSI = self.current_rx_power + (1 - self.filter_coeff)*self.average_RSSI

        delta = 0
        if self.average_RSSI > rx_target_high or self.average_RSSI < rx_target_low:
            ideal = rx_target - self.average_RSSI

            diffs = np.abs(ideal - self.DELTA_P_i)
            # diffs[diffs <= ideal] = 120

            index_of_min = np.argmin(diffs)
            delta = self.DELTA_P_i[index_of_min]

        return np.clip(self.current_tx_power + delta, MIN_TX_POWER, MAX_TX_POWER)

class Smith_2011(TPCMethodInterface):

    def __init__(self, nr_samples, K=3):
        super().__init__(nr_samples)
        Rx_sens = [-95, -93, -90, -86]
        self.RX = Rx_sens[K]
        self.levels_k = self.RX + np.arange(0, 47.5, 2.5)
        self.a = [7.5, 10, 7.5, 5][K]
        self.b = [1, 0, 1, 1][K]
    
    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high):
        mag_l = self.current_rx_power

        level_k_i = self.levels_k[0]
        for i, level in enumerate(self.levels_k[1:]):
            if self.levels_k[i-1] < mag_l <= self.levels_k[i]:
                level_k_i = level
                break

        C = self.RX + self.a

        if level_k_i <= C:
            ret = 0
        elif C + 2.5 <= level_k_i <= C + 25:
            ret = C + self.b - level_k_i
        else:
            ret = -25 + self.b
        return np.clip(ret, MIN_TX_POWER, MAX_TX_POWER)

class Kim(TPCMethodInterface):
    
    def __init__(self, nr_samples, N: int = 10, TRL: float = -88):
        super().__init__(nr_samples)
        self.n = N
        self.long_term_window_size = 5*N

        self.TRL = TRL
        self.delta = 0
        self.TRH = self.TRL + 3.0 + self.delta

        self.rssi_history_short = collections.deque(maxlen=N)
        self.rssi_history_long = collections.deque(maxlen=5*N)

        self.hi_history = collections.deque(maxlen=N)
        self.li_history = collections.deque(maxlen=N)

        self.i = N

    def update_internal(self):
        pass

    def update_delta(self):
        hist_long = np.array(self.rssi_history_long)
        self.R_avg = np.average(hist_long)
        self.delta = math.sqrt(
            (1 / 5 * self.n)
            * sum(hist_long - self.R_avg)
        )
        return self.delta


    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high):
        Ri = self.current_rx_power

        # Update RSSI histories
        self.rssi_history_long.append(Ri)
        self.rssi_history_short.append(Ri)

        if Ri > self.TRH:
            hi = Ri
        else:
            hi = self.TRH

        if Ri < self.TRL:
            li = Ri
        else:
            li = self.TRL
        
        if self.i > 1:
            # eq 5
            E_link = (
                sum( (self.TRH - hi) / self.n) 
                + sum( (self.TRL - li) / self.n)
            )
            self.current_tx_power += E_link
            self.i = self.n
            return self.current_tx_power
        else:
            # eq 6
            self.TRH = self.TRL + 3 + self.update_delta()
            self.i -= 1
            return self.current_tx_power

class Sodhro(TPCMethodInterface):
    def __init__(self, nr_samples):
        super().__init__(nr_samples)
        self.R_avg = 0
        self.R_lowest = -80; self.R_latest = -80
        self.N = 0 # n in eq 4 & 5 in Sodhro
        self.R_i_R_avg_sum = 0
        self.TRL = -88; self.TRH_var = 0

    def update_internal(self):
        self.R_lowest = self.R_latest 
        self.R_latest = self.current_rx_power
        ALPHA_1 = 1.0; ALPHA_2 = 0.4

        if self.R_latest > self.R_avg:  # Update R_avg eq. (1, 2)
            self.R_avg = self.R_latest + (1 - ALPHA_1) * self.R_lowest
        elif self.R_latest < self.R_avg:
            self.R_avg = self.R_latest + (1 - ALPHA_2) * self.R_lowest

        # Update TRH_var dynamically eq. (4, 5)
        self.R_i_R_avg_sum += self.current_rx_power - self.R_avg
        self.N += 1
        sigma = math.sqrt( (1 / self.N) * self.R_i_R_avg_sum )
        self.TRH_var = self.TRL + sigma

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high):
        self.update_internal()
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
