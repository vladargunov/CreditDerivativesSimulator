"""
Base_strategy module constructs an initial class for
any trading strategy
"""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

class BaseStrategy(ABC):
    """
    Base strategy class, any other
    strategy must inherit from it
    """

    @abstractmethod
    def train_model(self, train_data : pd.DataFrame) -> None:
        """
        Method is called to prepare a model for simulation
        by providing it the training data
        """
        raise NotImplementedError('Create training procedure!')

    @abstractmethod
    def trade(self, daily_data : dict) -> Optional[dict]:
        """
        The method is called each simulation step and receives
        a dictionary of prices, which then shoudl be used in
        construction of a portfolio, which must also be in the
        form of dictionary with keys as assets and fractions of
        the portfolio as values
        """
        raise NotImplementedError('Create strategy logic!')
