from abc import ABCMeta, abstractmethod
import numpy as np

class Agent(metaclass=ABCMeta):
    """Abstract Agent class"""

    @abstractmethod
    def _model_init(self):
        """
        Initializes parameters used by the model based on predefined settings
        """

        raise NotImplementedError

    @abstractmethod
    def remember(self, state, action, reward, next_state, done):
        """
        Adds the current Env to the memory
        """

        raise NotImplementedError

    @abstractmethod
    def act(self, state):
        """
        Chooses the action (Buy, Sell, Sit)
        """
        raise NotImplementedError