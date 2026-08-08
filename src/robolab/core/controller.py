import abc
import numpy as np

class Controller(abc.ABC): 
    """<Abstract base class for all controllers.>"""

    @abc.abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        """<Act on the observation to produce an action.>"""
        ...

    def reset(self) -> None: 
        """<Reset the controller to its intial state.>"""
        return None