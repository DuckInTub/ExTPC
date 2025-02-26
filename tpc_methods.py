from abc import ABC, abstractmethod

class TPC_method_interface(ABC):
    def __init__(self):
        # Define the parameters with their default types
        self.rx_powers: list[float] = []
        self.tx_powers: list[float] = []
        self.lost_frames: list[int] = []
        self.current_rx_power: float = 0.0
        self.current_tx_power: float = 0.0
        self.power_received_average: float = 0.0


    @abstractmethod
    def next_transmitt_power(self, current_received_power: float) -> None:
        """
        Calculate and update the next transmitted power based on the current received power.
        Must be implemented by any subclass.
        """
        pass

def Constant(TCP_method_interface):
    def next_transmitt_power(self, current_received_power: float) -> float:
        return 0 # dB

