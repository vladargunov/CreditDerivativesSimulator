"""
Template for MeanRevesion/MeanDiversion 
Strategy with trailing average and incremental 
portfolio changes
Author: vladargunov
"""

import pandas as pd
import numpy as np
import subprocess

class MeanBased(BaseStrategy):
    """
    This strategy template implements a simple
    mean reversion or diversion strategy based
    on prices of each asset over the trailing_period
    Then the portfolio of each asset is updated
    by daily_change if such change is allowed and
    maximum allocation of portfolio has not been
    reached
    """
    def __init__(self, trailing_period : int=100, daily_change : float=0.001,
                 reversion : bool=True):
        # Define lookback period in days
        self.trailing_period = trailing_period
        # Set daily change
        self.daily_change = daily_change

        # Flag defining reversion or diversion
        self.reversion = reversion

    def __repr__(self):
        _repr = "MeanReversion" if self.reversion else "MeanDiversion"
        _repr += f"(trailing_period={self.trailing_period}," +\
                 f"daily_change={self.daily_change:.5f})"
        return _repr
         

    def train_model(self, train_data):
        """
        Set lookback lenght and frequency
        """
        # Get lookback data
        self.lookback_data = train_data.iloc[-self.trailing_period:]

        # Self asset names
        self.asset_names = train_data.columns

        # Create an empty portfolio
        self.portfolio = {asset_name : 0 for asset_name in self.asset_names}

        # Determine starting average_prices
        self.average_prices = self.determine_average(train_data.iloc[-self.trailing_period:])

        # Create debug file
        subprocess.run(f"touch {repr(self)}_debug.txt".split())
        self.debug_file = open(f"{repr(self)}_debug.txt", "a")

        # Introduce step for debugging purposes
        self.step = 0

    def determine_average(self, data : pd.DataFrame):
        """
        Determines the means of data columns excluding nans
        values
        """
        means = np.nanmean(data, axis=0)

        average_prices = {asset : _mean for asset, _mean in zip(self.asset_names, means)}

        return average_prices

    def trade(self, daily_data : dict) -> dict:
        """
        Increase holdings of each stock by self.daily_change
        for each asset if reversion argument is true and 
        current price of asset is less than its trailing average,
        and decrease otherwise.

        If reversion argument is false, the decision to change portfolio
        is opposite 

        If maximum allocation is reached, then do nothing
        """
        # Place a recent date into self.lookback_data
        self.lookback_data = \
                    self.lookback_data.append(daily_data, ignore_index=True)
        self.lookback_data = self.lookback_data.iloc[1:]

        # Get portfolio changes in the current day
        # ------------------------------------------------
        change_portfolio = {}
        change_factor = 1 if self.reversion else -1
        for asset_name in self.asset_names:
            if daily_data[asset_name] >  self.average_prices[asset_name]:
                change_portfolio[asset_name] = -change_factor * self.daily_change
            else:
                change_portfolio[asset_name] = change_factor * self.daily_change
        # ------------------------------------------------

        # Change each asset weight if possible
        current_sum = sum([abs(self.portfolio[asset_name]) for asset_name in self.asset_names])
        for asset_name in self.asset_names:
            previous_asset_weight = self.portfolio[asset_name]
            new_asset_weight = self.portfolio[asset_name] + change_portfolio[asset_name]
            new_sum = current_sum + abs(new_asset_weight) - abs(previous_asset_weight)
            if new_sum < 1:
                self.portfolio[asset_name] = new_asset_weight
                current_sum = new_sum
            else:
                self.debug_file.write(f'Maximum allocation is reached at step {self.step}\n')
                self.debug_file.write('Portfolio is {self.portfolio}\n')
        # ------------------------------------------------

        # Update average prices
        self.average_prices = self.determine_average(self.lookback_data)

        # Update step
        self.step += 1

        return self.portfolio
