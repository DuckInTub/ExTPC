from abc import ABC, abstractmethod
import numpy as np
import math
import collections
from numpy.lib.stride_tricks import sliding_window_view
from collections import Counter

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

class Guo(TPCMethodInterface):

    def __init__(self, nr_samples, frame_path_losses):
        super().__init__(nr_samples)
        self.alpha = 0.2
        self.beta = 0.3
        self.num_states = 7

        self.current_tx_power = 0
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

    def update_internal(self):
        self.update_prediction_matrix()

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high):
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

class Isak(TPCMethodInterface):

    NR_SAMPLES = 4 # Number of samples for regression

    def __init__(self, nr_samples, packet_loss_threshhold):
        super().__init__(nr_samples)
        self.packet_loss_limit = packet_loss_threshhold
        self.latest_pathlosses = collections.deque([], maxlen=Isak.NR_SAMPLES) # FIFO length 8
        self.sum_high = 0
        self.sum_low = -90
        self.iter = 0

    def update_internal(self):
        current_path_loss = self.current_rx_power - self.current_tx_power
        self.iter += 1
        self.latest_pathlosses.append(current_path_loss)
        self.sum_high += max(self.latest_pathlosses)
        self.sum_low  += min(self.latest_pathlosses)
        # print(self.sum_high / self.iter, self.sum_low / self.iter)

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

class Xiao_aggressive_2008(TPCMethodInterface):
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

class Xiao_conservative_2008(TPCMethodInterface):
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
    
class Xiao_aggressive_2009(TPCMethodInterface):
    def __init__(self, nr_of_samples, avg_weight_d=0.2, avg_weight_u=0.8):
        super().__init__(nr_of_samples)
        self.avg_weight_d = avg_weight_d # α_d in Xiaos paper
        self.avg_weight_u = avg_weight_u # α_u in Xiaos paper
        self.exp_avg_rx_power: float = 0.0 # R̅ in Xiaos paper

    def update_internal(self):
        if self.current_rx_power <= self.exp_avg_rx_power:
            self.exp_avg_rx_power = self.avg_weight_d*self.current_rx_power+(1-self.avg_weight_d)*self.exp_avg_rx_power
        else:
            self.exp_avg_rx_power = self.avg_weight_u*self.current_rx_power+(1-self.avg_weight_u)*self.exp_avg_rx_power

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high) -> float:
        if self.exp_avg_rx_power > rx_target_high:
            delta = -2
        if self.exp_avg_rx_power < rx_target_low:
            delta = 3
        if rx_target_low <= self.exp_avg_rx_power <= rx_target_high:
            delta = 0
        return np.clip(self.current_tx_power + delta, MIN_TX_POWER, MAX_TX_POWER)
    
class Xiao_balanced_2009(TPCMethodInterface):
    def __init__(self, nr_of_samples, avg_weight_d=0.8, avg_weight_u=0.8):
        super().__init__(nr_of_samples)
        self.avg_weight_d = avg_weight_d # α_d in Xiaos paper
        self.avg_weight_u = avg_weight_u # α_u in Xiaos paper
        self.exp_avg_rx_power: float = 0.0 # R̅ in Xiaos paper

    def update_internal(self):
        if self.current_rx_power <= self.exp_avg_rx_power:
            self.exp_avg_rx_power = self.avg_weight_d*self.current_rx_power+(1-self.avg_weight_d)*self.exp_avg_rx_power
        else:
            self.exp_avg_rx_power = self.avg_weight_u*self.current_rx_power+(1-self.avg_weight_u)*self.exp_avg_rx_power

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high) -> float:
        if self.exp_avg_rx_power > rx_target_high:
            delta = -1
        if self.exp_avg_rx_power < rx_target_low:
            delta = 3
        if rx_target_low <= self.exp_avg_rx_power <= rx_target_high:
            delta = 0
        return np.clip(self.current_tx_power + delta, MIN_TX_POWER, MAX_TX_POWER)

class Xiao_conservative_2009(TPCMethodInterface):
    def __init__(self, nr_of_samples, avg_weight_d=0.8, avg_weight_u=0.2):
        super().__init__(nr_of_samples)
        self.avg_weight_d = avg_weight_d # α_d in Xiaos paper
        self.avg_weight_u = avg_weight_u # α_u in Xiaos paper
        self.exp_avg_rx_power: float = 0.0 # R̅ in Xiaos paper

    def update_internal(self):
        if self.current_rx_power <= self.exp_avg_rx_power:
            self.exp_avg_rx_power = self.avg_weight_d*self.current_rx_power+(1-self.avg_weight_d)*self.exp_avg_rx_power
        else:
            self.exp_avg_rx_power = self.avg_weight_u*self.current_rx_power+(1-self.avg_weight_u)*self.exp_avg_rx_power

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
        self.DELTA_P_i = np.array([-3, -2, -1, 0, 1, 2, 3, 4])

    def update_internal(self):
        self.average_RSSI = self.filter_coeff*self.current_rx_power + (1 - self.filter_coeff)*self.average_RSSI

    def next_transmitt_power(self, rx_target, rx_target_low, rx_target_high) -> float:
        delta = 0
        if self.average_RSSI > rx_target_high or self.average_RSSI < rx_target_low:
            ideal = rx_target - self.average_RSSI

            diffs = np.abs(ideal - self.DELTA_P_i)

            index_of_min = np.argmin(diffs)
            delta = self.DELTA_P_i[index_of_min]

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