from abc import ABC, abstractmethod

class PowerInterface(ABC):
    def __init__(self):
        # Define the parameters with their default types
        self.power_transmitted: list[float] = []
        self.power_received: list[float] = []
        self.power_current: float = 0.0

    @abstractmethod
    def next_transmitt_power(self, current_received_power: float) -> None:
        """
        Calculate and update the next transmitted power based on the current received power.
        Must be implemented by any subclass.
        """
        pass